import os
import pprint

import gymnasium as gym
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


class PacmanRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PacmanRewardWrapper, self).__init__(env)
        self.last_position = None
        self.stationary_steps = 0
        self.initial_position = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_position = None
        self.stationary_steps = 0
        self.initial_position = (obs[10], obs[16])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        player_x = obs[10]
        player_y = obs[16]
        current_position = (player_x, player_y)

        if self.last_position is not None and current_position == self.last_position:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0

        self.last_position = current_position

        if self.stationary_steps > 30 and current_position != self.initial_position:
            reward -= 5

        return obs, reward, terminated, truncated, info


def make_env():
    env = gym.make("ALE/MsPacman-ram-v5")
    env = PacmanRewardWrapper(env)
    return env


DEVICE = "cuda:2"
env = make_env()
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
train_envs = SubprocVectorEnv([make_env for _ in range(10)])
test_envs = SubprocVectorEnv([make_env for _ in range(10)])

# Define Q-network
q_net = Net(
    state_shape, action_shape, hidden_sizes=[1024, 1024, 1024], device=DEVICE
).to(DEVICE)
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
    if env.spec.reward_threshold:
        return mean_rewards >= env.spec.reward_threshold
    return False


# Set checkpoint paths
ckpt = None
save_path = "/root/gym/rl_compexp/save/MsPacMan-DQN1024"


def train_dqn():
    print(save_path)
    print(policy)

    def save_best_fn(policy: DQNPolicy):
        torch.save(policy.state_dict(), os.path.join(save_path, "dqn_best.pth"))

    if ckpt:
        pth = torch.load(os.path.join(ckpt, "dqn_best.pth"))
        policy.load_state_dict(pth)

    log_path = "log/DQN_MsPacman"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=5000,
        step_per_epoch=100000,
        step_per_collect=1000,
        episode_per_test=10,
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
