import os
import pprint

import gymnasium as gym
from gymnasium.wrappers import TransformObservation
import tianshou as ts
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def make_env():
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env = TransformObservation(env, lambda obs: list(obs), observation_space=None)
    return env


DEVICE = "cuda:0"
env = make_env()
state_shape = 3
action_shape = env.action_space.shape or env.action_space.n
train_envs = SubprocVectorEnv([make_env for _ in range(10)])
test_envs = SubprocVectorEnv([make_env for _ in range(10)])

# Define Q-network
q_net = Net(state_shape, action_shape, hidden_sizes=[64] * 2, device=DEVICE).to(DEVICE)
optim = torch.optim.Adam(q_net.parameters(), lr=2.5e-4)
scheduler = StepLR(optim, step_size=2000, gamma=0.98)

policy = DQNPolicy(
    model=q_net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.99,
    estimation_step=3,
    target_update_freq=500,
).to(DEVICE)

# Experience replay buffer
buffer = VectorReplayBuffer(100000, buffer_num=len(train_envs), ignore_obs_next=True)
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=False)


def stop_fn(mean_rewards: float) -> bool:
    return mean_rewards >= 0.2


# Set checkpoint paths
# ckpt = "/root/gym/rl_compexp/save/Blackjack2-DQN64"
ckpt = None

save_path = "/root/gym/rl_compexp/save/Blackjack2-DQN64"
os.makedirs(save_path, exist_ok=True)


def train_dqn():
    print(save_path)
    print(policy)

    def save_best_fn(policy: DQNPolicy):
        torch.save(policy.state_dict(), os.path.join(save_path, "dqn_best.pth"))

    if ckpt:
        pth = torch.load(os.path.join(ckpt, "dqn_best.pth"))
        policy.load_state_dict(pth)

    log_path = "log/DQN_Blackjack"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=5000,
        step_per_epoch=100000,
        step_per_collect=100,
        episode_per_test=10000,
        batch_size=256,
        update_per_step=0.1,
        test_fn=None,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    torch.save(policy.state_dict(), os.path.join(save_path, "dqn.pth"))
    pprint.pprint(result)
    writer.close()


train_dqn()
