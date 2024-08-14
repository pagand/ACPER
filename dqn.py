import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import arenax_minigames


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

# Initialize wandb
run = wandb.init(
    project="dqn-squidhunt",
    entity="marslab",
    sync_tensorboard=True,
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 10000,
        "env_name": "SquidHunt-v0"
    }
)

# Initialize the Squid Hunt environment
# env = gym.make('SquidHunt-v0')
env = CustomSquidHuntEnv()

# Define the hyperparameters for DQN
dqn_params = {
    'learning_rate': 1e-3,
    'buffer_size': 10000,
    'learning_starts': 1000,
    'batch_size': 64,
    'gamma': 0.99,
    'train_freq': 4,
    'target_update_interval': 1000,
    'exploration_fraction': 0.1,
    'exploration_final_eps': 0.02,
    'max_grad_norm': 10,
    'policy_kwargs': dict(net_arch=[256, 256])
}

# Create a DQN model
model = DQN('MlpPolicy', env, verbose=1, device="cpu", **dqn_params, tensorboard_log=f"runs/{run.id}")

# Train the model
model.learn(total_timesteps=500000, callback=WandbCallback(gradient_save_freq=100,
                                                        model_save_path=f"models/{run.id}",
                                                        verbose=2,))

# create env for evaluation
vec_env = model.get_env()
# obs = vec_env.reset()
# model.predict(obs)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

# Save the final model
model.save("dqn_squid_hunt")

# Finish the wandb run
run.finish()

# To test the trained model
sum_reward = 0
for _ in range(10):
    obs = vec_env.reset()
    creward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        creward += reward
    sum_reward += creward

print(f"Average reward: {sum_reward / 10}")