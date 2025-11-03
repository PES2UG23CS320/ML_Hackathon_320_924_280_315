# dqn_agent.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, hidden=1024, output_dim=26):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(s2), np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim, lr=1e-4, gamma=0.99, tau=1e-3, hidden=1024):

        self.net = DQN(input_dim, hidden=hidden).to(DEVICE)
        self.target = DQN(input_dim, hidden=hidden).to(DEVICE)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.replay = ReplayBuffer()
        self.loss_fn = nn.MSELoss()

    def act(self, state_np, epsilon=0.1, guessed_mask=None):
        # state_np: 1D numpy
        if random.random() < epsilon:
            # choose random unguessed
            allowed = [i for i in range(26) if (guessed_mask is None or guessed_mask[i] == 0)]
            return random.choice(allowed)
        else:
            self.net.eval()
            with torch.no_grad():
                s = torch.from_numpy(state_np).float().unsqueeze(0).to(DEVICE)
                q = self.net(s).cpu().numpy().ravel()
                # mask guessed letters
                if guessed_mask is not None:
                    q = q - 1e6 * guessed_mask  # heavily penalize guessed ones
                return int(np.argmax(q))

    def push(self, *args):
        self.replay.push(*args)

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None
        s, a, r, s2, done = self.replay.sample(batch_size)
        s = torch.from_numpy(s).float().to(DEVICE)
        s2 = torch.from_numpy(s2).float().to(DEVICE)
        a = torch.from_numpy(a).long().to(DEVICE)
        r = torch.from_numpy(r).float().to(DEVICE)
        done = torch.from_numpy(done).float().to(DEVICE)

        q_values = self.net(s)  # B x A
        q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            q_target = r + self.gamma * q_next * (1.0 - done)

        loss = self.loss_fn(q_a, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # soft update
        for param, target_param in zip(self.net.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target.load_state_dict(self.net.state_dict())
