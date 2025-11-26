import os
import warnings
import gymnasium as gym
import random
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = "cuda:1"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(seed)


def make_env():
    return gym.make("LunarLander-v2", continuous=True)


train_envs = SubprocVectorEnv([make_env for _ in range(8)])
test_envs = SubprocVectorEnv([make_env for _ in range(8)])

state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].shape

act_net = Net(state_shape, hidden_sizes=[1024] * 2, device=DEVICE)
critic_net = Net(state_shape, hidden_sizes=[1024] * 2, device=DEVICE)
actor = ActorProb(act_net, action_shape, unbounded=True, device=DEVICE).to(DEVICE)
critic = Critic(critic_net, device=DEVICE).to(DEVICE)

actor_critic = ActorCritic(actor, critic)
torch.nn.init.constant_(actor.sigma_param, -0.5)
for m in actor_critic.modules():
    if isinstance(m, torch.nn.Linear):
        # orthogonal initialization
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(m.bias)
# do last policy layer scaling, this will make initial actions have (close to)
# 0 mean and std, and will help boost performances,
# see https://arxiv.org/abs/2006.05990, Fig.24 for details
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=lambda loc, scale: Independent(Normal(loc, scale), 1),
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=True,
    action_space=train_envs.action_space[0],
    action_scaling=True,  # 对连续动作进行缩放
    deterministic_eval=True,
).to(DEVICE)

# train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 8))
# test_collector = Collector(policy, test_envs)
train_collector = Collector(policy, train_envs)
test_collector = Collector(policy, test_envs)


stop_fn = lambda mean_rewards: mean_rewards >= 270
ckpt = None
save_path = r"/root/gym/rl_compexp/save/LunarLander-Continuous-PPO256"
os.makedirs(save_path, exist_ok=True)


def save_best_fn(policy: PPOPolicy):
    state = {"model": policy.state_dict(), "optim": policy.optim.state_dict()}
    torch.save(state, os.path.join(save_path, "ppo_best.pth"))


if ckpt:
    pth = torch.load(os.path.join(ckpt, "ppo_best.pth"))
    print(pth)
    policy.load_state_dict(pth["model"])

result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=5000,
    step_per_epoch=51200,
    repeat_per_collect=5,
    episode_per_test=100,
    batch_size=64,
    step_per_collect=512,
    test_fn=None,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
).run()

# 保存模型
state = {
    "model": policy.state_dict(),
    "optim": policy.optim.state_dict(),
}
torch.save(state, os.path.join(save_path, "ppo.pth"))
