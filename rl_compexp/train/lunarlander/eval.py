import os
import torch
import gymnasium as gym
import numpy as np
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, Batch
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy
import matplotlib.pyplot as plt
def evaluate_policy():
    # Device configuration
    DEVICE = "cuda:0"
    
    # Create environment
    env = gym.make("LunarLander-v3")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    print(env.observation_space)
    # Create Q-network
    net = Net(state_shape, hidden_sizes=[64] * 2, action_shape=action_shape, device=DEVICE)
    q_net = net.to(DEVICE)
    
    # Create optimizer (needed for policy creation)
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    
    # Create policy
    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
        reward_normalization=False,
        is_double=True,
        action_space=env.action_space,
    ).to(DEVICE)
    
    # Load saved model
    save_path = "/root/gym/rl_compexp/save/LunarLander-DQN64"
    checkpoint = torch.load(os.path.join(save_path, "dqn.pth"))
    policy.load_state_dict(checkpoint["model"])
    
    # Evaluation
    policy.eval()
    rewards = []
    lengths = []
    # Collect all states
    all_states = []
    for episode in range(10):
        episode_reward = 0
        steps = 0
        done = False
        state, _ = env.reset()
        
        while not done:
            all_states.append(state)
            batch = Batch(obs=state.reshape(1, -1), info={})
            action = policy.forward(batch).act.item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1}/100: Reward = {episode_reward:.2f}, Length = {steps}")
    
    # Print statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Reward above 250: {np.sum(np.array(rewards) >= 250)}")

    all_states = np.array(all_states)

    # Plot feature distributions
    num_features = all_states.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features))
    for i in range(num_features):
        axes[i].hist(all_states[:, i], bins=50, alpha=0.75)
        axes[i].set_title(f"Feature {i + 1} Distribution")
    plt.tight_layout()
    #save fig
    plt.savefig('feature_distribution.png')
    # max value of features
    max_features = np.max(all_states, axis=0)
    min_features = np.min(all_states, axis=0)
    print(f"Max values of features: {max_features}")
    print(f"Min values of features: {min_features}")
    env.close()

def save_hidden_outputs():
    hidden_outputs = []
    def hook_fn(module, input, output):
        # Assuming output is a tensor, detach and move to CPU
        hidden_outputs.append(output.detach().cpu().numpy())

    # Device configuration
    DEVICE = "cuda:0"
    
    # Create environment
    env = gym.make("LunarLander-v3")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    
    # Create Q-network
    net = Net(state_shape, hidden_sizes=[64] * 2, action_shape=action_shape, device=DEVICE)
    q_net = net.to(DEVICE)
    
    # Create optimizer (needed for policy creation)
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    
    # Create policy
    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
        reward_normalization=False,
        is_double=True,
        action_space=env.action_space,
    ).to(DEVICE)
    # policy.model = Net(
    #   (model): MLP(
    #     (model): Sequential(
    #       (0): Linear(in_features=8, out_features=64, bias=True)
    #       (1): ReLU()
    #       (2): Linear(in_features=64, out_features=64, bias=True)
    #       (3): ReLU()
    #       (4): Linear(in_features=64, out_features=4, bias=True)
    #     )
    #   )
    # )
    policy.model.model.model[2].register_forward_hook(hook_fn)
    # Load saved model
    save_path = "/root/gym/rl_compexp/save/LunarLander-DQN64"
    checkpoint = torch.load(os.path.join(save_path, "dqn.pth"))
    policy.load_state_dict(checkpoint["model"])
    
    # Evaluation
    policy.eval()
    states=[]
    actions=[]
    # Collect all states
    all_states = []
    for episode in range(500):
        steps = 0
        done = False
        state, _ = env.reset()
        
        while not done:
            all_states.append(state)
            batch = Batch(obs=state.reshape(1, -1), info={})
            action = policy.forward(batch).act.item()
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps>1e5:
                break
        if steps>1e5:
            break
    states=np.array(states[:10000])
    actions=np.array(actions[:10000])
    hidden_outputs=np.array(hidden_outputs[:10000])
    print(states.shape, actions.shape, hidden_outputs.shape)
    np.save(save_path+'/states.npy', states)
    np.save(save_path+'/actions.npy', actions)
    np.save(save_path+'/hidden_outputs.npy', hidden_outputs)
        
    
def get_weights():
    # Device configuration
    DEVICE = "cuda:0"
    
    # Create environment
    env = gym.make("LunarLander-v3")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    
    # Create Q-network
    net = Net(state_shape, hidden_sizes=[64] * 2, action_shape=action_shape, device=DEVICE)
    q_net = net.to(DEVICE)
    
    # Create optimizer (needed for policy creation)
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    
    # Create policy
    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
        reward_normalization=False,
        is_double=True,
        action_space=env.action_space,
    ).to(DEVICE)
    
    # Load saved model
    save_path = "/root/gym/rl_compexp/save/LunarLander-DQN64"
    checkpoint = torch.load(os.path.join(save_path, "dqn.pth"))
    policy.load_state_dict(checkpoint["model"])
    
    # Get weights
    weights = policy.model.model.model[4].weight.t().detach().cpu().numpy()
    print(weights.shape)
    np.save(save_path+'/weights.npy', weights)

if __name__ == "__main__":
    evaluate_policy()
    # save_hidden_outputs()
    # get_weights()