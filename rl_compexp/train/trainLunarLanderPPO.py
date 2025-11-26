import os
import warnings
import gymnasium as gym
import torch
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

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = "cuda:1"


def make_env():
    return gym.make("LunarLander-v3")


train_envs = SubprocVectorEnv([make_env for _ in range(8)])
test_envs = SubprocVectorEnv([make_env for _ in range(8)])

state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].n

net = Net(state_shape, hidden_sizes=[64] * 2, device=DEVICE)
actor = Actor(net, action_shape, device=DEVICE).to(DEVICE)
critic = Critic(net, device=DEVICE).to(DEVICE)

actor_critic = ActorCritic(actor, critic).to(DEVICE)
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
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
).to(DEVICE)
print(policy.actor)
# policy.actor.preprocess.model.model

# train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 8))
# test_collector = Collector(policy, test_envs)

# # 创建 Tensorboard logger
# # writer = SummaryWriter("log/ppo")
# # logger = TensorboardLogger(writer)
# stop_fn = lambda mean_rewards: mean_rewards >= 270
# ckpt = "/root/gym/rl_compexp/save/LunarLander-PPO1024"
# # ckpt = None
# save_path = "../save/LunarLander-PPO1024"


# def save_best_fn(policy: PPOPolicy):
#     # 保存actor和critic的状态字典
#     state = {"model": policy.state_dict(), "optim": policy.optim.state_dict()}
#     torch.save(state, os.path.join(save_path, "ppo_best.pth"))


# if ckpt:
#     pth = torch.load(os.path.join(ckpt, "ppo_best.pth"))
#     print(pth)
#     policy.load_state_dict(pth["model"])

# result = OnpolicyTrainer(
#     policy=policy,
#     train_collector=train_collector,
#     test_collector=test_collector,
#     max_epoch=5000,  # 设置最大epoch数
#     step_per_epoch=10000,  # 每个epoch的步数
#     repeat_per_collect=2,
#     episode_per_test=100,
#     batch_size=1024,
#     step_per_collect=256,
#     test_fn=None,
#     stop_fn=stop_fn,  # 传递停止函数
#     save_best_fn=save_best_fn,  # 传递保存函数
#     # logger=logger,
# ).run()

# # print(f"Finished training! Use {result.train_episode} episodes, ")

# # 保存模型
# state = {
#     "model": policy.state_dict(),
#     "optim": policy.optim.state_dict(),
# }
# torch.save(state, os.path.join(save_path, "ppo.pth"))
