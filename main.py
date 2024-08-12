import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import arenax_minigames
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb


# init wandb
wandb.init(project="acer-squidhunt")


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
        self.avg_reward = (self.avg_reward * (len(self.buffer_priority) -1) + reward )/ len(self.buffer_priority)

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
actor_lr = 0.0001 #0.0007
critic_lr = 0.0001 #0.001
gamma = 0.8 # 0.99
buffer_size = 5000# 1000
batch_size = 32 #32
trust_region_constraint = 1.0
entropy_coeff = 0.3# 0.1 # premature convergence to suboptimal policies
max_grad_norm = 0.5 #0.5
num_updates = 3000


# learnign rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)



# Environment and ACER Initialization
env = gym.make("SquidHunt-v0")
#env = gym.make("CartPole-v1")

# buffer = ReplayBuffer(buffer_size)
buffer = PrioritizedReplayBuffer(buffer_size)
network = ActorCriticNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
actor_optimizer = optim.Adam(network.actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(network.critic.parameters(), lr=critic_lr)


# reward for each episode
reward_list = []

print_freq = num_updates
count_priority = 0

# Training Loop
for episode in range(num_updates):
    state, _ = env.reset()
    done = False
    creward = 0
    cactorloss = 0
    ccriticloss = 0
    counter = 0
    while not done:
        # Select action and store in buffer
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, value = network(state_tensor)

        action = torch.multinomial(policy, 1).item()
        log_prob = network.get_log_probs(state_tensor, torch.tensor([action]))
        
        next_state, reward, done, truncated, _ = env.step(action)
        # reward shaping to be between -1 and 1
        reward = 1 if reward > 1 else reward
        reward = -1 if reward < -1 else reward

        # only add to the buffer if the transition has a higher reward than the average reward of the buffer
        buffer.add(state, action, reward, next_state, done, log_prob)
        if reward >= buffer.avg_reward:
            buffer.add_priority(state, action, reward, next_state, done, log_prob)
            count_priority += 1
        if done:
            buffer.add_done(state, action, reward, next_state, done, log_prob)

        if len(buffer) >= batch_size:
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
                       "entropy_loss": entropy_loss.item(), "log_probs": log_probs.mean().item(), "total_loss": total_loss.item(),
                       "priority_buffer": count_priority, "avg_reward": buffer.avg_reward})

            nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)

            actor_optimizer.step()
            critic_optimizer.step()
            cactorloss += actor_loss.item()
            ccriticloss += critic_loss.item()
            counter += 1

        state = next_state

        # commuliative reward
        creward += reward
    reward_list.append(creward)
    # wandb log for each epoch
    if counter:
        wandb.log({"epoch":episode + 1, "reward": creward, "avg_actor_loss": cactorloss/counter, "avg_critic_loss": ccriticloss/counter})

    # print reward every print_freq episodes
    if episode % print_freq == 0:
        print(f"Episode {episode + 1}, Reward: {reward_list[-1]}")

env.close()

# Plot the rewards
plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.show()


# wait
input("Press Enter to end...")
