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
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = "cuda:0"

def make_env():
    return gym.make("LunarLander-v3")

# 创建训练和测试环境
train_envs = SubprocVectorEnv([make_env for _ in range(8)])
test_envs = SubprocVectorEnv([make_env for _ in range(8)])

# 获取状态空间和动作空间的维度
state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].n
print(state_shape, action_shape)
# 创建Q网络
net = Net(state_shape, hidden_sizes=[64] * 2, action_shape=action_shape, device=DEVICE)
q_net = net.to(DEVICE)


# 创建优化器
optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)

# 创建DQN策略
policy = DQNPolicy(
    model=q_net,
    optim=optim,
    discount_factor=0.99,
    estimation_step=3,
    target_update_freq=500,
    reward_normalization=False,
    is_double=True,
    action_space=train_envs.action_space[0],
    # target_model=target_q_net
).to(DEVICE)
print(policy)
# 创建收集器
buffer_size = 20000
train_collector = Collector(
    policy, 
    train_envs, 
    VectorReplayBuffer(buffer_size, 8)
)
test_collector = Collector(policy, test_envs)

# 定义停止条件和保存函数
stop_fn = lambda mean_rewards: mean_rewards >= 270
save_path = "/root/gym/rl_compexp/save/LunarLander-DQN64"
if not os.path.exists(save_path):
    os.makedirs(save_path,exist_ok=True)

def save_best_fn(policy):
    state = {
        "model": policy.state_dict(),
        "optim": policy.optim.state_dict()
    }
    torch.save(state, os.path.join(save_path, "dqn_best.pth"))

# 加载检查点（如果存在）
ckpt_path = None  # 设置检查点路径
if ckpt_path and os.path.exists(ckpt_path):
    checkpoint = torch.load(os.path.join(ckpt_path, "dqn_best.pth"))
    policy.load_state_dict(checkpoint["model"])
    policy.optim.load_state_dict(checkpoint["optim"])

# 开始训练
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
).run()

# 保存最终模型
state = {
    "model": policy.state_dict(),
    "optim": policy.optim.state_dict()
}
torch.save(state, os.path.join(save_path, "dqn.pth"))