import warnings
import gymnasium as gym
import torch
import numpy as np
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tqdm import tqdm
import tianshou

warnings.filterwarnings("ignore", category=DeprecationWarning)


DEVICE = "cuda:1"
# 定义环境
env = gym.make("LunarLander-v2")
state_shape = env.observation_space.shape
action_shape = env.action_space.n
net = Net(state_shape, hidden_sizes=[1024] * 2, device=DEVICE)
actor = Actor(net, action_shape, device=DEVICE)
critic = Critic(net, device=DEVICE)


def make_env():
    return gym.make("LunarLander-v2")


test_envs = SubprocVectorEnv([make_env for _ in range(25)])
dist_fn = torch.distributions.Categorical

policy = PPOPolicy(
    actor=actor,
    critic=critic,
    dist_fn=dist_fn,
    optim=None,
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=True,
    action_space=env.action_space,
    action_scaling=False,
    deterministic_eval=True,
).to(DEVICE)
# 加载预训练模型
policy.load_state_dict(torch.load("../save/LunarLander-PPO1024/ppo.pth")["model"])


test_collector = Collector(policy, test_envs)
policy.eval()
result = test_collector.collect(n_episode=1000)


# def testLander(env, agent, episodes=100):
#     total_reward = 0
#     successful_episodes = 0
#     for i in tqdm(range(episodes)):
#         state, info = env.reset()
#         episode_reward = 0
#         cnt = 0
#         episode_success = False
#         while cnt < 1000:
#             action = policy.forward(
#                 tianshou.data.Batch(obs=state.reshape(1, -1), info="")
#             ).act.item()
#             state, reward, done, _, _ = env.step(action)
#             episode_reward += reward
#             cnt += 1
#             if done:
#                 if episode_reward >= 200:  # 假设奖励大于等于200视为成功着陆
#                     episode_success = True
#                 break
#         total_reward += episode_reward
#         if episode_success:
#             successful_episodes += 1
#     average_reward = total_reward / episodes
#     success_rate = successful_episodes / episodes * 100
#     print(f"Average reward over {episodes} episodes: {average_reward}")
#     print(
#         f"Successful episodes: {successful_episodes}/{episodes} ({success_rate:.2f}%)"
#     )


print(np.sum(result.returns > 200))
# testLander(env, policy)
