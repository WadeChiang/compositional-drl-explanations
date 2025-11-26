import os
import warnings
import gymnasium as gym
import torch
import pprint
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import StepLR


warnings.filterwarnings("ignore", category=DeprecationWarning)


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


DEVICE = "cuda:1"
env = make_env()
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
train_envs = SubprocVectorEnv([make_env for _ in range(10)])
test_envs = SubprocVectorEnv([make_env for _ in range(10)])

net = Net(state_shape, hidden_sizes=[1024] * 3, device=DEVICE)
actor = Actor(net, action_shape, device=DEVICE).to(DEVICE)
critic = Critic(net, device=DEVICE).to(DEVICE)

actor_critic = ActorCritic(actor, critic).to(DEVICE)
optim = torch.optim.Adam(actor_critic.parameters(), lr=2.5e-4)
scheduler = StepLR(
    optim, step_size=2000, gamma=0.98
)  # 每100个step，学习率减少为原来的0.9
dist_fn = torch.distributions.Categorical
policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist_fn,
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=True,
    action_space=train_envs.action_space[0],
    action_scaling=False,
    deterministic_eval=True,
    lr_scheduler=scheduler,
).to(DEVICE)

buffer = VectorReplayBuffer(
    100000,
    buffer_num=len(train_envs),
    ignore_obs_next=True,
    save_only_last_obs=False,
)
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=False)


def stop_fn(mean_rewards: float) -> bool:
    if env.spec.reward_threshold:
        return mean_rewards >= env.spec.reward_threshold
    return False


# ckpt = "/root/gym/rl_compexp/save/MsPacMan-PPO1024"
ckpt = None
save_path = "/root/gym/rl_compexp/save/MsPacMan-PPO1024"


def train():
    print(save_path)

    def save_best_fn(policy: PPOPolicy):
        torch.save(policy.state_dict(), os.path.join(save_path, "ppo_best.pth"))

    if ckpt:
        pth = torch.load(os.path.join(ckpt, "ppo_best.pth"))
        policy.load_state_dict(pth)

    log_path = "log/PPO_MsPacman"
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    print(policy)
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=5000,
        step_per_epoch=100000,
        repeat_per_collect=2,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        test_fn=None,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    torch.save(policy.state_dict(), os.path.join(save_path, "ppo.pth"))
    pprint.pprint(result)
    writer.close()


def eval():
    policy.load_state_dict(torch.load(os.path.join(save_path, "ppo_best.pth")))
    policy.eval()
    result = test_collector.collect(n_episode=20)
    pprint.pprint(result)


def sample():
    policy.load_state_dict(torch.load(os.path.join(save_path, "ppo_best.pth")))
    policy.eval()
    result = train_collector.collect(n_step=50000)
    obs = buffer.obs[:50000]
    actions = buffer.act[:50000]
    np.save("obs.npy", obs)
    np.save("actions.npy", actions)


train()
