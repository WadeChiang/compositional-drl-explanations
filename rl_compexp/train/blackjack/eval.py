import os
import torch
import numpy as np
import gymnasium as gym
from tianshou.data import Collector
from tianshou.utils.net.common import Net
from gymnasium.wrappers import TransformObservation
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
import pprint


# Function to create the environment (same as in training)
def make_env():
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env = TransformObservation(env, lambda obs: list(obs), observation_space=None)
    return env


# Load the trained model and policy
DEVICE = "cuda:2"
env = make_env()
state_shape = 3
action_shape = env.action_space.shape or env.action_space.n

# Load the trained Q-network (the saved model)
q_net = Net(state_shape, action_shape, hidden_sizes=[64] * 2, device=DEVICE).to(DEVICE)
optim = torch.optim.Adam(
    q_net.parameters(), lr=2.5e-4
)  # Same optimizer used during training
policy = DQNPolicy(
    model=q_net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.99,
    estimation_step=3,
    target_update_freq=500,
).to(DEVICE)

# Load the trained policy weights
checkpoint_path = "/root/gym/rl_compexp/save/Blackjack-DQN64/dqn_best.pth"
policy.load_state_dict(torch.load(checkpoint_path))

# Create the test environment
test_envs = SubprocVectorEnv([make_env for _ in range(10)])

# Create a Collector for the test environment (no exploration noise for evaluation)
test_collector = Collector(policy, test_envs, exploration_noise=False)


# Define the evaluation function
def evaluate_dqn():
    # Run the model on the test environment for 100 episodes
    result = test_collector.collect(n_episode=1000)
    pprint.pprint(result)
    print(np.count_nonzero(result.returns == 1))
    print(np.count_nonzero(result.returns == 0))
    print(np.count_nonzero(result.returns == -1))


# Evaluate the model
evaluate_dqn()
