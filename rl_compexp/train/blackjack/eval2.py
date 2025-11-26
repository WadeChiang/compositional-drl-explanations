import os
import torch
import numpy as np
import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.utils.net.common import Net
from gymnasium.wrappers import TransformObservation
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
import pprint


def make_env():
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env = TransformObservation(env, lambda obs: np.array(obs), observation_space=None)
    return env


DEVICE = "cuda:0"
env = make_env()
state_shape = 3
action_shape = env.action_space.shape or env.action_space.n

q_net = Net(state_shape, action_shape, hidden_sizes=[64] * 2, device=DEVICE).to(DEVICE)
optim = torch.optim.Adam(q_net.parameters(), lr=2.5e-4)
policy = DQNPolicy(
    model=q_net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.99,
    estimation_step=3,
    target_update_freq=500,
).to(DEVICE)
policy.eval()
checkpoint_path = "/root/gym/rl_compexp/save/Blackjack2-DQN64/dqn_best.pth"
policy.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))

hidden_outputs = []

print(policy)
print(action_shape)
def hook_fn(module, input, output):
    # Assuming output is a tensor, detach and move to CPU
    hidden_outputs.append(output.detach().cpu().numpy())


policy.model.model.model[2].register_forward_hook(hook_fn)

weight = policy.model.model.model[4].weight.detach().cpu().t().numpy()
print("weight", weight.shape)


def evaluate_dqn_and_save_obs():
    observations = []
    actions = []
    global hidden_outputs
    while len(observations) < 10000:
        obs, _ = env.reset()
        done = False
        while not done:
            observations.append(obs)
            act = policy(Batch(obs=obs.reshape(1, -1), info="")).act.item()
            actions.append(act)
            obs, reward, done, _, _ = env.step(act)
    print(len(observations), len(actions), len(hidden_outputs))
    observations = observations[:10000]
    actions = actions[:10000]
    hidden_outputs = hidden_outputs[:10000]
    observations = np.array(observations)
    actions = np.array(actions)
    hidden_outputs = np.array(hidden_outputs).reshape(-1,64)
    save_dir = "/root/gym/rl_compexp/save/Blackjack2-DQN64"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "states.npy"), observations)
    np.save(os.path.join(save_dir, "actions.npy"), actions)
    np.save(os.path.join(save_dir, "hidden_outputs.npy"), hidden_outputs)


# Run evaluation and save observations
evaluate_dqn_and_save_obs()
np.save("/root/gym/rl_compexp/save/Blackjack2-DQN64/weights.npy", weight)
