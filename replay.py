from collections import namedtuple
import random
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._num_added = 0
        self.buffer = []
        self.steps_done = 0  # Initialize steps_done

    def add(self, state, action, next_state, reward):
        if reward is None:
            reward = 0
        if len(self.buffer) >= self._capacity:
            self.buffer.pop(0)
        # Ensure action is scalar
        if isinstance(action, torch.Tensor):
            action = action.item()
        # Ensure states are numpy arrays with consistent shapes
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32) if next_state is not None else np.zeros_like(state)
        self.buffer.append(Transition(state, action, next_state, reward))
        self._num_added += 1
        self.steps_done += 1  # Increment steps_done

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
