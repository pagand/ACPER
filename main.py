import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import arenax_minigames
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb


# init wandb
wandb.init(project="acer-squidhunt")

class CustomSquidHuntEnv(gym.Env):
    def __init__(self, **kwargs):
        super(CustomSquidHuntEnv, self).__init__()
        self.env = gym.make("SquidHunt-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        # Call the original step function
        state, reward, done, truncated, info = self.env.step(action)
        
        # Implement reward shaping or other modifications here
        if reward < -1:
            reward = -1
        elif reward < 0.4:
            reward = 0.1

        return state, reward, done, truncated, info

    def reset(self, seed=None):
            return self.env.reset(seed=seed)

    def render(self, mode="human"):
        return self.env.render(mode=mode)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.avg_reward = 0

    def add(self, state, action, reward, next_state, done, log_prob):
        # Wrap all components into a tuple and add to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob)
        self.position = (self.position + 1) % self.capacity
        self.avg_reward = (self.avg_reward * (len(self.buffer) -1) + reward )/ len(self.buffer)

    def sample(self, batch_size):
        # Randomly sample a batch of experiences
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, log_prob = zip(*batch)
        return state, action, reward, next_state, done, log_prob

    def __len__(self):
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, ratio=0.2):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.ratio = ratio
        self.buffer_priority = []
        self.position_priority = 0
        self.buffer_done = []
        self.position_done = 0
        self.avg_reward = 0
        

    def add(self, state, action, reward, next_state, done, log_prob):
        # Wrap all components into a tuple and add to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob)
        self.position = (self.position + 1) % self.capacity
        
    def add_priority(self, state, action, reward, next_state, done, log_prob):
        # Wrap all components into a tuple and add to the buffer
        if len(self.buffer_priority) < self.capacity:
            self.buffer_priority.append(None)
        self.buffer_priority[self.position_priority] = (state, action, reward, next_state, done, log_prob)
        self.position_priority = (self.position_priority + 1) % self.capacity
        # self.avg_reward = (self.avg_reward * (len(self.buffer_priority) -1) + reward )/ len(self.buffer_priority)
        self.avg_reward = (self.avg_reward * (self.capacity -1) + reward )/ self.capacity

    def add_done(self, state, action, reward, next_state, done, log_prob):
        # Wrap all components into a tuple and add to the buffer
        if len(self.buffer_done) < self.capacity:
            self.buffer_done.append(None)
        self.buffer_done[self.position_done] = (state, action, reward, next_state, done, log_prob)
        self.position_done = (self.position_done + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences
        batch = random.sample(self.buffer, int(batch_size*(1-self.ratio)))
        if len(self.buffer_done) < batch_size*self.ratio/2: 
            batch_priority = random.sample(self.buffer_priority, int(batch_size*self.ratio))
        else:
            batch_done = random.sample(self.buffer_done, int(batch_size*self.ratio/2))
            batch_priority = random.sample(self.buffer_priority, int(batch_size*self.ratio/2))
            batch.extend(batch_done)
        batch.extend(batch_priority)
        state, action, reward, next_state, done, log_prob = zip(*batch)
        return state, action, reward, next_state, done, log_prob

    def __len__(self):
        return len(self.buffer)



# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.1), # edited
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        policy = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy, value
    
    def get_log_probs(self, states, actions):
        action_probs, _ = self.forward(states)
        action_log_probs = torch.log(action_probs)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1))
        return selected_log_probs.squeeze(1)




# Importance Sampling and Trust Region Update
def compute_importance_sampling_ratios(policy_net, states, actions, old_log_probs):
    """
    Compute the importance sampling ratios for the actions taken in the past.
    
    :param policy_net: The current policy network.
    :param states: The states stored in the replay buffer.
    :param actions: The actions stored in the replay buffer.
    :param old_log_probs: The log probabilities of the actions under the old policy.
    :return: The importance sampling ratios.
    """
    # Get the log probabilities of the actions under the current policy
    new_log_probs = policy_net.get_log_probs(states, actions)
    
    # Compute the importance sampling ratios
    ratios = torch.exp(new_log_probs - old_log_probs)
    
    return ratios

def trust_region_update(ratios, q_values, constraint=1.0):
    scaled_ratios = torch.clamp(ratios, 1.0 - constraint, 1.0 + constraint)
    return scaled_ratios * q_values


# Hyperparameters
actor_lr = 0.00008 #0.00008
critic_lr = 0.0005 #0.0005
gamma = 0.9 # 0.9
buffer_size = 10000# 5000
batch_size = 32 #32
trust_region_constraint = 1.0
entropy_coeff = 0.1# 0.1 # premature convergence to suboptimal policies
max_grad_norm = 0.5 #0.5
total_timesteps = 100000
epoch = 5 # per step
retrain_model = False

# save in the model name
model_name = f"alr:{actor_lr}, clr:{critic_lr}, gamma:{gamma}, buffer:{buffer_size}, bsize:{batch_size}, trc:{trust_region_constraint}, entropy:{entropy_coeff}, maxgrad:{max_grad_norm}, timesteps:{total_timesteps}, epoch:{epoch}"


# learnign rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)



# Environment and ACER Initialization
# env = gym.make("SquidHunt-v0")
#env = gym.make("CartPole-v1")
env = CustomSquidHuntEnv()

# buffer = ReplayBuffer(buffer_size)
buffer = PrioritizedReplayBuffer(buffer_size)
# if retrain is true, load the model
network = ActorCriticNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
if retrain_model:
    network.load_state_dict(torch.load("model.pth"))
    print("***********************loaded the model***********************")
else:
    print("new model")

actor_optimizer = optim.Adam(network.actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(network.critic.parameters(), lr=critic_lr)



count_priority = 0
current_timesteps = 0

# Training Loop
while True:
    if current_timesteps >= total_timesteps: # no more training
            print("***********reached the total timesteps***********")
            break
    state, _ = env.reset()
    done = False
    creward = 0
    cactorloss = 0
    ccriticloss = 0
    counter = 0
    while not done:
        current_timesteps += 1
        if current_timesteps >= total_timesteps: # no more training
            break
        # Select action and store in buffer
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, value = network(state_tensor)

        action = torch.multinomial(policy, 1).item()
        log_prob = network.get_log_probs(state_tensor, torch.tensor([action]))
        
        next_state, reward, done, truncated, _ = env.step(action)

        # commuliative reward
        creward += reward

        # only add to the buffer if the transition has a higher reward than the average reward of the buffer
        buffer.add(state, action, reward, next_state, done, log_prob)
        if reward >= buffer.avg_reward:
            buffer.add_priority(state, action, reward, next_state, done, log_prob)
            count_priority += 1
        if done:
            buffer.add_done(state, action, reward, next_state, done, log_prob)

        if len(buffer) >= batch_size:
            for _ in range(epoch): # epoch
                states, actions, rewards, next_states, dones, old_log_probs = buffer.sample(batch_size)

                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.int64)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)
                old_log_probs_tensor = torch.stack(old_log_probs).squeeze(1)  


                policy_new, value_new = network(states_tensor)
                _, value_next = network(next_states_tensor)

                q_values = rewards_tensor + gamma * value_next.squeeze(1) * (1 - dones_tensor)

                ratios = compute_importance_sampling_ratios(network, states_tensor, actions_tensor, old_log_probs_tensor)
                adjusted_q_values = trust_region_update(ratios, q_values, trust_region_constraint)

                # Critic loss (Mean Squared Error)
                critic_loss = nn.MSELoss()(value_new.squeeze(1), adjusted_q_values.detach())

                # Actor loss (Policy Gradient with Importance Sampling and Entropy Regularization)
                log_probs = torch.log(policy_new.gather(1, actions_tensor.unsqueeze(-1)))
                actor_loss = -(log_probs * adjusted_q_values.detach()).mean()
                entropy_loss = -(policy_new * log_probs).mean()
                # total_actor_loss = actor_loss - entropy_coeff * entropy_loss
                total_actor_loss = actor_loss + entropy_coeff * entropy_loss
                
                total_loss = critic_loss + total_actor_loss

                # Backpropagation and Optimization
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                # critic_loss.backward(retain_graph=True)
                # total_actor_loss.backward()
                total_loss.backward()

                # log to wandb
                wandb.log({"total_actor_loss":total_actor_loss.item() ,"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "ratio": ratios.mean().item(), 
                        "entropy_loss": entropy_loss.item(), "log_probs": log_probs.mean().item(),
                        "priority_buffer": count_priority, "avg_reward": buffer.avg_reward, "current_timesteps": current_timesteps})

                nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)

                actor_optimizer.step()
                critic_optimizer.step()
                cactorloss += actor_loss.item()
                ccriticloss += critic_loss.item()
                counter += 1

        state = next_state

        
    # wandb log for each epoch
    if counter:
        wandb.log({"reward": creward, "avg_actor_loss": cactorloss/counter, "avg_critic_loss": ccriticloss/counter})

# save the model
torch.save(network.state_dict(), f"{model_name}.pth")


# save the model with .onnx format for competition
env = gym.make("SquidHunt-v0")
env.save(network, f"model_name", 'pytorch', use_onnx=True)



# To test the trained model
sum_reward = 0
for _ in range(10):
    state, _ = env.reset()
    creward = 0
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, _ = network(state_tensor)
        # action = torch.multinomial(policy, 1).item()
        # chose the argmax action
        action = torch.argmax(policy).item()
        state, reward, done, truncated, _ = env.step(action)
        creward += reward
    sum_reward += creward

print(f"Average reward: {sum_reward / 10}")
env.close()
wandb.finish()