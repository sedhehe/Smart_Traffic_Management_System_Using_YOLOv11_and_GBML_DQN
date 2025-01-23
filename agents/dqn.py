import random
import torch
import torch.nn as nn
import networks
from collections import namedtuple
import copy
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DqnAgent:
    def __init__(
        self,
        mode: str,
        replay,
        target_update: int,
        gamma: float,
        use_sgd: bool,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        input_dim: int,
        output_dim: int,
        batch_size: int,
        network_file: str = ''
    ):
        self.mode = mode
        self.replay = replay
        self.target_update = target_update
        self.gamma = gamma
        self.use_sgd = use_sgd
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = output_dim
        self.batch_size = batch_size

        self.network_file = network_file
        self.policy_net = networks.DqnNetwork(input_dim, output_dim).to(device)
        self.target_net = networks.DqnNetwork(input_dim, output_dim).to(device)
        self.policy_net_copy = networks.DqnNetwork(input_dim, output_dim).to(device)
        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=torch.device(device)))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_steps = 0
        self.z = None
        self.fixed_gamma = copy.deepcopy(gamma)
        self.update_gamma = False
        self.q_value_batch_avg = 0

    def select_action(self, state, steps_done, invalid_action):
        original_state = state
        state = torch.from_numpy(state).to(device)
        if self.mode == 'train':
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            if sample > eps_threshold:
                with torch.no_grad():
                    _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                    if invalid_action:
                        action = sorted_indices[1].item()
                    else:
                        action = sorted_indices[0].item()
            else:
                decrease_state = [(original_state[0] + original_state[4]) / 2,
                                  (original_state[1] + original_state[5]) / 2,
                                  (original_state[2] + original_state[6]) / 2,
                                  (original_state[3] + original_state[7]) / 2]
                congest_phase = [i for i, s in enumerate(decrease_state) if abs(s - 1) < 1e-2]
                if len(congest_phase) > 0 and not invalid_action:
                    action = random.choice(congest_phase)
                else:
                    action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                if invalid_action:
                    action = sorted_indices[1].item()
                else:
                    action = sorted_indices[0].item()

        # Ensure action is within the valid range
        action = max(0, min(action, self.n_actions - 1))
        return action

    def learn(self):
        if self.mode == 'train':
            if self.replay.steps_done <= 10000:
                return
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025) if self.use_sgd else torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025)

            transitions = self.replay.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # Convert numpy arrays to tensors
            state_batch = torch.tensor(np.array(batch.state), device=device, dtype=torch.float32)
            action_batch = torch.tensor(np.array(batch.action), device=device, dtype=torch.long).view(self.batch_size, 1)
            reward_batch = torch.tensor(np.array(batch.reward), device=device, dtype=torch.float32).view(self.batch_size, 1)
            next_state_batch = torch.tensor(np.array([s for s in batch.next_state if s is not None]), device=device, dtype=torch.float32)

            # Debugging shapes
            # print(f"state_batch shape: {state_batch.shape}")
            # print(f"action_batch shape: {action_batch.shape}")
            # print(f"next_state_batch shape: {next_state_batch.shape}")
            # print(f"reward_batch shape: {reward_batch.shape}")

            # Ensure next_state_batch is not empty
            if next_state_batch.size(0) == 0:
                return

            # Compute Q-values
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute next state values
            with torch.no_grad():
                argmax_action = self.policy_net(next_state_batch).max(1)[1].view(-1, 1)
                q_max = self.target_net(next_state_batch).gather(1, argmax_action)
                expected_state_action_values = reward_batch + self.gamma * q_max

            # Loss calculation
            loss = loss_fn(state_action_values, expected_state_action_values)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
            self.learn_steps += 1
            self.update_gamma = True

    def cal_z(self, state_batch, action_batch, q_max):
        self.policy_net_copy.load_state_dict(self.policy_net.state_dict())
        z_optimizer = torch.optim.SGD(self.policy_net_copy.parameters(), lr=0.0001)
        state_action_copy_values = self.policy_net_copy(state_batch).gather(1, action_batch)
        z_optimizer.zero_grad()
        f_gamma_grad = torch.mean(0.00025 * q_max * state_action_copy_values)
        f_gamma_grad.backward()

        # Ensure gradients are not None
        self.z = {
            'l1.weight': self.policy_net_copy.l1.weight.grad if self.policy_net_copy.l1.weight.grad is not None else torch.zeros_like(self.policy_net_copy.l1.weight),
            'l1.bias': self.policy_net_copy.l1.bias.grad if self.policy_net_copy.l1.bias.grad is not None else torch.zeros_like(self.policy_net_copy.l1.bias),
            'l2.weight': self.policy_net_copy.l2.weight.grad if self.policy_net_copy.l2.weight.grad is not None else torch.zeros_like(self.policy_net_copy.l2.weight),
            'l2.bias': self.policy_net_copy.l2.bias.grad if self.policy_net_copy.l2.bias.grad is not None else torch.zeros_like(self.policy_net_copy.l2.bias)
        }

    def learn_gamma(self):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(self.batch_size, 1)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            q_max = self.target_net(next_state_batch).gather(1, argmax_action)
            expected_state_action_values = reward_batch + self.fixed_gamma * q_max

        loss = loss_fn(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()

        l1_weight = self.policy_net.l1.weight.grad * self.z['l1.weight']
        l1_bias = self.policy_net.l1.bias.grad * self.z['l1.bias']
        l2_weight = self.policy_net.l2.weight.grad * self.z['l2.weight']
        l2_bias = self.policy_net.l2.bias.grad * self.z['l2.bias']

        gamma_grad = -0.99 * torch.mean(torch.cat((l1_weight.view(-1), l1_bias.view(-1), l2_weight.view(-1), l2_bias.view(-1))))
        self.gamma += gamma_grad
        self.update_gamma = False
