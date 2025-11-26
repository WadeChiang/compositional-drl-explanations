import warnings

import gymnasium as gym
import numpy as np
import tianshou
import torch
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from feature_extract import HookLayer
from settings import DEVICE

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
## LunarLander
PPOPolicy(
  (actor): Actor(
    (preprocess): Net(
      (model): MLP(
        (model): Sequential(
          (0): Linear(in_features=8, out_features=1024, bias=True)
          (1): ReLU()
          (2): Linear(in_features=1024, out_features=1024, bias=True)
          (3): ReLU()
        )
      )
    )
    (last): MLP(
      (model): Sequential(
        (0): Linear(in_features=1024, out_features=4, bias=True)
      )
    )
  )
"""


def get_policy(ckpt_path, env_name):
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    net = Net(state_shape, hidden_sizes=[1024] * 2, device=DEVICE)
    actor = Actor(net, action_shape, device=DEVICE)
    critic = Critic(net, device=DEVICE)
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
    policy.load_state_dict(
        torch.load("../save/LunarLander-PPO1024/ppo.pth", map_location=DEVICE)["model"]
    )
    hook = HookLayer()
    print(policy.actor)
    hook.hook_layer(policy.actor.preprocess.model.model[2])
    # hook.hook_layer(policy.actor.last.model[0])
    return policy, hook


def get_PPO(ckpt_path, data_path, env_name="LunarLander-v3"):
    # 定义环境
    policy, hook = get_policy(ckpt_path, env_name)
    policy.eval()
    data = np.load(data_path)
    res = policy(tianshou.data.Batch(obs=data, info=""))
    # print(res)
    # print(hook.features_blobs[0].shape, len(hook.features_blobs))
    """
    Batch(
    logits: tensor([[2.7141e-02, 5.1383e-03, 1.6782e-03, 9.6604e-01],
                    [3.0328e-02, 6.5162e-03, 2.0768e-03, 9.6108e-01],
                    [3.3722e-02, 8.3266e-03, 2.5748e-03, 9.5538e-01],
                    ...,
                    [3.1087e-02, 9.2603e-01, 4.2027e-02, 8.5140e-04],
                    [2.9454e-02, 9.1168e-01, 5.8071e-02, 7.9296e-04],
                    [2.7847e-02, 8.8497e-01, 8.6443e-02, 7.4081e-04]], device='cuda:3',
                   grad_fn=<SoftmaxBackward0>),
        act: tensor([3, 3, 3,  ..., 1, 1, 1], device='cuda:3'),
        state: None,
        dist: Categorical(probs: torch.Size([10000, 4])),
    )
    (10000, 1024) 1
    """
    output = res.logits.cpu().detach().numpy()
    inputs = data
    feature = hook.features_blobs[0]
    weight = policy.actor.last.model[0].weight.t().cpu().detach().numpy()
    return inputs, feature, output, weight, policy


def get_io(ckpt_path, data_path, env_name):
    # 定义环境
    policy, _ = get_policy(ckpt_path, env_name)
    data = np.load(data_path)
    res = policy(tianshou.data.Batch(obs=data, info=""))
    # print(res)
    # print(hook.features_blobs[0].shape, len(hook.features_blobs))
    """
    Batch(
    logits: tensor([[2.7141e-02, 5.1383e-03, 1.6782e-03, 9.6604e-01],
                    [3.0328e-02, 6.5162e-03, 2.0768e-03, 9.6108e-01],
                    [3.3722e-02, 8.3266e-03, 2.5748e-03, 9.5538e-01],
                    ...,
                    [3.1087e-02, 9.2603e-01, 4.2027e-02, 8.5140e-04],
                    [2.9454e-02, 9.1168e-01, 5.8071e-02, 7.9296e-04],
                    [2.7847e-02, 8.8497e-01, 8.6443e-02, 7.4081e-04]], device='cuda:3',
                   grad_fn=<SoftmaxBackward0>),
        act: tensor([3, 3, 3,  ..., 1, 1, 1], device='cuda:3'),
        state: None,
        dist: Categorical(probs: torch.Size([10000, 4])),
    )
    (10000, 1024) 1
    """
    actions = res.act.cpu().detach().numpy()
    outputs = actions
    inputs = data
    return inputs, outputs


if __name__ == "__main__":
    ckpt_path = "../save/LunarLander-PPO1024/ppo.pth"
    data_path = "../save/lunar.npy"
    env_name = "LunarLander-v2"
    inputs, outputs = get_io(ckpt_path, data_path, env_name)
    print(inputs[0:5])
