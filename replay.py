from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._num_added = 0
        self._storage = [None] * capacity

    def add(self, state, next_state, reward, action):
        if reward is not None:
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            action = torch.tensor(action).unsqueeze(0).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
            self._storage[self._num_added % self._capacity] = Transition(state, action, next_state, reward)
            self._num_added += 1

    def sample(self, batch_size=1):
        indices = np.random.randint(0, self.size, batch_size)
        return [self._storage[i] for i in indices]

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return min(self._num_added, self._capacity)

    @property
    def steps_done(self):
        return self._num_added

    @property
    def storage(self):
        return self._storage