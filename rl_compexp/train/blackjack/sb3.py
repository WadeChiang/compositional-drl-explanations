import os
import gymnasium as gym
import warnings
import torch
import pandas as pd
import numpy as np
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces


class BlackjackTupleToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 将观察空间从Tuple转换为Box
        self.observation_space = spaces.Box(
            low=0,
            high=max(32, 11, 2),  # 使用最大可能值作为上限
            shape=(3,),  # 三个特征：玩家点数、庄家明牌、是否有可用的A
            dtype=np.float32,
        )

    def observation(self, obs):
        # 将tuple观察转换为numpy数组
        return np.array(obs, dtype=np.float32)


def make_env():
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env = BlackjackTupleToArrayWrapper(env)
    return env


def train():
    num_envs = 8
    # 创建向量化环境
    env = make_vec_env(
        make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv, monitor_dir="./logs/"
    )

    DEVICE = "cuda:2"
    # 定义模型参数
    model_params = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 250,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "device": DEVICE,
    }

    # 创建DQN模型
    model = DQN(env=env, **model_params)

    # 创建checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="blackjack_dqn",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 训练模型
    total_timesteps = 1000000
    model.learn(
        total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True
    )

    # 保存最终模型
    model.save("blackjack_dqn_final")

    # 评估模型
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100, deterministic=True
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 关闭环境
    env.close()


if __name__ == "__main__":
    train()
