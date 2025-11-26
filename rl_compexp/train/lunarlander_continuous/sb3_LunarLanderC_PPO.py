import os
import gym
import warnings
import torch
import pandas as pd
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm

warnings.filterwarnings("ignore", category=Warning)

# Set proxy if needed
os.environ["HTTP_PROXY"] = "http://127.0.0.1:17893"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:17893"

# Load the pretrained model
checkpoint = load_from_hub(
    repo_id="sb3/ppo-LunarLanderContinuous-v2",
    filename="ppo-LunarLanderContinuous-v2.zip",
)
model = PPO.load(checkpoint)

# Create the environment
env = gym.make("LunarLanderContinuous-v2")

# Ensure the model is in evaluation mode
model.policy.eval()

# Dictionary to store the hidden layer outputs
hidden_outputs = []

# Lists to store observations and actions
observations = []
actions = []


# Function to register as a forward hook
def hook_fn(module, input, output):
    # Assuming output is a tensor, detach and move to CPU
    hidden_outputs.append(output.detach().cpu().numpy())


policy_net = model.policy.mlp_extractor.policy_net

# Register hook on the second last layer
# For example, if it's a Sequential, access the appropriate layer
# Here, we assume it's a Sequential model
target_layer = policy_net[-1]
hook_handle = target_layer.register_forward_hook(hook_fn)

# Reset the lists
hidden_outputs = []
observations = []
actions = []

# Custom evaluation loop to capture data
num_eval_episodes = 400
for episode in tqdm(range(num_eval_episodes)):
    obs, _ = env.reset()
    done = False
    while not done:
        # Convert observation to tensor
        with torch.no_grad():
            action, _states = model.predict(obs, deterministic=True)
        # Save observation and action
        observations.append(obs)
        actions.append(action)
        # Take action in the environment
        obs, reward, done, _, info = env.step(action)

# Remove the hook
hook_handle.remove()

# Create a DataFrame to store the data
data = pd.DataFrame(
    {
        "observation": observations,
        "action": actions,
        "hidden_output": hidden_outputs,
    }
)

# Save the data to a file, e.g., CSV
data.to_csv("evaluation_data.csv", index=False)

print(f"Saved {len(observations)} steps of data to 'evaluation_data.csv'.")


# Optionally, evaluate the policy and print rewards
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
