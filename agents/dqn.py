import random
import torch
import torch.nn as nn
import networks
from collections import namedtuple
import copy
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DqnAgent:
    def __init__(self, mode, replay_buffer, target_update_interval, gamma, use_sgd, epsilon_start, epsilon_end, epsilon_decay, input_dim, output_dim, batch_size, network_file=''):
        self.mode = mode
        self.replay_buffer = replay_buffer
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.use_sgd = use_sgd
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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
        state = torch.from_numpy(state)
        if self.mode == 'train':
            sample = random.random()
            epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * steps_done / self.epsilon_decay)
            if sample > epsilon_threshold:
                with torch.no_grad():
                    _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                    return sorted_indices[1] if invalid_action else sorted_indices[0]
            else:
                decrease_state = [(state[0] + state[4]) / 2, (state[1] + state[5]) / 2, (state[2] + state[6]) / 2, (state[3] + state[7]) / 2]
                congested_phase = [i for i, s in enumerate(decrease_state) if abs(s-1) < 1e-2]
                if congested_phase and not invalid_action:
                    return random.choice(congested_phase)
                else:
                    return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                return sorted_indices[1] if invalid_action else sorted_indices[0]

    def learn(self):
        if self.mode == 'train' and self.replay_buffer.steps_done > 10000:
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025) if self.use_sgd else torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025)

            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).view(self.batch_size, 1)
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
                q_max = self.target_net(next_state_batch).gather(1, argmax_action)
                expected_state_action_values = reward_batch + self.gamma * q_max
                self.q_value_batch_avg = torch.mean(state_action_values).item()

            loss = loss_fn(state_action_values, expected_state_action_values)
            optimizer.zero_grad()
            loss.backward()

            self.calculate_z(state_batch, action_batch, q_max)

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            self.learn_steps += 1
            self.update_gamma = True

    def calculate_z(self, state_batch, action_batch, q_max):
        self.policy_net_copy.load_state_dict(self.policy_net.state_dict())
        z_optimizer = torch.optim.SGD(self.policy_net_copy.parameters(), lr=0.0001)
        state_action_copy_values = self.policy_net_copy(state_batch).gather(1, action_batch)
        z_optimizer.zero_grad()
        f_gamma_grad = torch.mean(0.00025 * q_max * state_action_copy_values)
        f_gamma_grad.backward()
        self.z = {'layer1.weight': self.policy_net_copy.layer1.weight.grad,
                  'layer1.bias': self.policy_net_copy.layer1.bias.grad,
                  'layer2.weight': self.policy_net_copy.layer2.weight.grad,
                  'layer2.bias': self.policy_net_copy.layer2.bias.grad}

    def learn_gamma(self):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)

        transitions = self.replay_buffer.sample(self.batch_size)
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

        layer1_weight = self.policy_net.layer1.weight.grad * self.z['layer1.weight']
        layer1_bias = self.policy_net.layer1.bias.grad * self.z['layer1.bias']
        layer2_weight = self.policy_net.layer2.weight.grad * self.z['layer2.weight']
        layer2_bias = self.policy_net.layer2.bias.grad * self.z['layer2.bias']

        gamma_grad = -0.99 * torch.mean(torch.cat((layer1_weight.view(-1), layer1_bias.view(-1), layer2_weight.view(-1), layer2_bias.view(-1))))
        self.gamma += gamma_grad
        self.update_gamma = False