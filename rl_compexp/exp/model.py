import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

DQN_DEVICE = torch.device("cuda:2")


class QNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=4):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out

class QNet512(nn.Module):   
    def __init__(self, input_size=8, hidden_size=512, output_size=4):
        super(QNet512, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out

class QNet1024(nn.Module):
    def __init__(self, input_size=8, hidden_size=1024, output_size=4):
        super(QNet1024, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out


class DQN():
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3, eval=True, ckpt=None, QNet = QNet):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        # model
        if ckpt:
            print(f"Loading checkpoint {ckpt}")
            self.net_eval = QNet(input_size=n_states,
                                 output_size=n_actions).to(DQN_DEVICE)
            self.net_eval.load_state_dict(torch.load(ckpt))
            self.net_target = QNet(input_size=n_states,
                                   output_size=n_actions).to(DQN_DEVICE)
            self.net_target.load_state_dict(torch.load(ckpt))
        else:
            self.net_eval = QNet(input_size=n_states,
                                 output_size=n_actions).to(DQN_DEVICE)
            self.net_target = QNet(input_size=n_states,
                                   output_size=n_actions).to(DQN_DEVICE)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.scheduler = scheduler.ExponentialLR(self.optimizer, gamma = 0.95)
        self.criterion = nn.MSELoss()

        # memory
        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0    # update cycle counter
        self.eval = eval

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DQN_DEVICE)

        with torch.no_grad():
            action_values = self.net_eval(state)
            # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
        return action

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                if self.counter % (self.learn_step * 10000) == 0:
                    self.scheduler.step()
                    print(self.scheduler.get_last_lr()[0])
    # Raw DQN learn
    # def learn(self, experiences):
    #     states, actions, rewards, next_states, dones = experiences

    #     q_target = self.net_target(next_states).detach().max(axis=1)[
    #         0].unsqueeze(1)
    #     # target, if terminal then y_j = rewards
    #     y_j = rewards + self.gamma * q_target * (1 - dones)
    #     q_eval = self.net_eval(states).gather(1, actions)

    #     # loss backprop
    #     loss = self.criterion(q_eval, y_j)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # soft update target network
    #     self.softUpdate()

    # Double DQN learn
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: Use the eval network to select actions
        q_eval_next = self.net_eval(next_states).detach()
        _, best_actions = q_eval_next.max(1, keepdim=True)

        # Double DQN: Use the target network to evaluate the action's Q-value
        q_target_next = self.net_target(next_states).detach().gather(1, best_actions)
        y_j = rewards + self.gamma * q_target_next * (1 - dones)

        q_eval = self.net_eval(states).gather(1, actions)

        # Loss and backpropagation steps remain the same
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.softUpdate()
    
    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(
                self.tau*eval_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(DQN_DEVICE)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(DQN_DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(DQN_DEVICE)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(DQN_DEVICE)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DQN_DEVICE)

        return (states, actions, rewards, next_states, dones)
