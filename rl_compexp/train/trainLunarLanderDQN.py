from tqdm import trange
import gymnasium as gym
import numpy as np
import torch
import sys
sys.path.append("..")
from exp import model


def testLander_no_visual(env, agent, episodes, file, episode):
    total_reward = 0
    for i in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        cnt = 0
        while cnt < 1000:
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            cnt += 1
            if done:
                break
        total_reward += episode_reward
    average_reward = total_reward / episodes
    file.write(
        f"Episode: {episode}, Eval reward over 100 episode: {average_reward}\n")
    file.flush()
    return average_reward


def train(env, agent, n_episodes=2000, max_steps=300, eps_start=1.0, eps_end=0.1, eps_decay=0.999, target=200, ckpt=False):
    score_hist = []
    epsilon = eps_start
    log = open("train.log", "w")
    eval_log = open("eval.log", 'w')
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    # bar_format = '{l_bar}{bar:10}{r_bar}'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    best_score_avg = -np.inf  # Initialize best score average
    for idx_epi in pbar:
        state, info = env.reset()
        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        score_hist.append(score)
        score_avg = np.mean(score_hist[-100:])
        epsilon = max(eps_end, epsilon*eps_decay)

        pbar.set_postfix_str(
            f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}")
        pbar.update(0)
        if (idx_epi % 2**8 == 0):
            log.write(
                f"Episode: {idx_epi}, Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}\n")
            log.flush()
        if (idx_epi % 2**12 == 0):
            eval_reward = testLander_no_visual(env, agent, episodes=500,
                                               file=eval_log, episode=idx_epi)
        # Early stop
        if len(score_hist) >= 100:
            if eval_reward >= target:
                break
        # Save best checkpoint
        if eval_reward > best_score_avg:
            best_score_avg = eval_reward
            if ckpt is not None:
                torch.save(agent.net_eval.state_dict(),
                           f'../save/best-ckpt-{ckpt}.pth')

    if (idx_epi+1) < n_episodes:
        print("\nTarget Reached!")
    else:
        print("\nDone!")

    if ckpt is not None:
        torch.save(agent.net_eval.state_dict(), f'../save/ckpt-{ckpt}.pth')

    return score_hist


BATCH_SIZE = 256
LR = 1e-3
EPISODES = 2000000
TARGET_SCORE = 260.     # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 10000     # max memory buffer size
LEARN_STEP = 256          # how often to learn
TAU = 5e-3              # for soft update of target parameters
SAVE_CKPT = "mlp1024-260"      # save trained network .pth file
device = "cuda:2"
# load_ckpt= './save/ckpt-mlp64-250.pth'
load_ckpt = None


env = gym.make('LunarLander-v2')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
agent = model.DQN(
    n_states=num_states,
    n_actions=num_actions,
    batch_size=BATCH_SIZE,
    lr=LR,
    gamma=GAMMA,
    mem_size=MEMORY_SIZE,
    learn_step=LEARN_STEP,
    tau=TAU,
    ckpt=load_ckpt,
    eval=False,
    QNet=model.QNet1024
)
score_hist = train(env, agent, n_episodes=EPISODES,
                   target=TARGET_SCORE, ckpt=SAVE_CKPT)
