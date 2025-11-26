import os
import warnings
import gymnasium as gym
import torch
import numpy as np
import multiprocessing as mp
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

def train_dqn(config_name, hidden_sizes):
    DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    
    def make_env():
        return gym.make("LunarLander-v3")
    
    # Create training and test environments
    train_envs = SubprocVectorEnv([make_env for _ in range(8)])
    test_envs = SubprocVectorEnv([make_env for _ in range(8)])
    
    try:
        # Get state and action space dimensions
        state_shape = train_envs.observation_space[0].shape
        action_shape = train_envs.action_space[0].n
        
        print(f"\nTraining DQN with hidden sizes: {hidden_sizes}")
        
        # Create save directory for this configuration
        save_path = f"/root/gym/rl_compexp/save/LunarLander-DQN-{config_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # Create Q network
        net = Net(state_shape, hidden_sizes=hidden_sizes, action_shape=action_shape, device=DEVICE)
        q_net = net.to(DEVICE)
        
        # Create optimizer
        optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        
        # Create DQN policy
        policy = DQNPolicy(
            model=q_net,
            optim=optim,
            discount_factor=0.99,
            estimation_step=3,
            target_update_freq=500,
            reward_normalization=False,
            is_double=True,
            action_space=train_envs.action_space[0],
        ).to(DEVICE)
        
        # Create collectors
        buffer_size = 20000
        train_collector = Collector(
            policy, 
            train_envs, 
            VectorReplayBuffer(buffer_size, 8)
        )
        test_collector = Collector(policy, test_envs)
        
        # Define stop condition and save function
        stop_fn = lambda mean_rewards: mean_rewards >= 270
        
        def save_best_fn(policy):
            state = {
                "model": policy.state_dict(),
                "optim": policy.optim.state_dict()
            }
            torch.save(state, os.path.join(save_path, "dqn_best.pth"))
        
        # Start training
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
        
        # Save final model
        state = {
            "model": policy.state_dict(),
            "optim": policy.optim.state_dict()
        }
        torch.save(state, os.path.join(save_path, "dqn.pth"))
        
        print(f"Finished training {config_name}")
        
    finally:
        # Clean up
        train_envs.close()
        test_envs.close()

if __name__ == "__main__":
    # Define different hidden layer configurations
    hidden_configs = {
        "128x2": [128] * 2,
        "256x2": [256] * 2,
        "512x2": [512] * 2,
        "64x3": [64] * 3,
        "128x3": [128] * 3,
        "256x3": [256] * 3,
        "512x3": [512] * 3,
        "64x4": [64] * 4,
        "128x4": [128] * 4,
        "256x4": [256] * 4,
        "512x4": [512] * 4
    }
    
    # Create processes
    processes = []
    for config_name, hidden_sizes in hidden_configs.items():
        p = mp.Process(target=train_dqn, args=(config_name, hidden_sizes))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()