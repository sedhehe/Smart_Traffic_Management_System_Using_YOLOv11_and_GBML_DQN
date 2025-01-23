import gym
import os
import sys
import random
from environment.traffic_signal import TrafficSignal
from replay import ReplayBuffer

import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib


class SumoEnv(gym.Env):
    def __init__(
        self,
        net_file: str,
        route_file: str,
        skip_range: int,
        simulation_time: float,
        yellow_time: int,
        delta_rs_update_time: int,
        reward_fn: str,
        mode: str,
        use_gui: bool = False
    ):
        self._net = net_file
        self._route = route_file
        self.skip_range = skip_range
        self.simulation_time = simulation_time
        self.yellow_time = yellow_time

        self.reward_fn = reward_fn
        self.mode = mode
        self.use_gui = use_gui
        self.train_state = None
        self.next_state = None
        self.last_phase_state = None
        self.change_action_time = None
        self.sumo = None
        # temp
        self.queue = []
        self.avg_queue = []
        self.avg_wait = []
        self.total_rewards = 0  # Initialize as an integer

        self.sumoBinary = 'sumo'
        if self.use_gui:
            self.sumoBinary = 'sumo-gui'

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        conn = traci
        self.ts_id = traci.trafficlight.getIDList()[0]
        self.traffic_signal = TrafficSignal(
            ts_id=self.ts_id,
            yellow_time=self.yellow_time,
            simulation_time=simulation_time,
            delta_rs_update_time=delta_rs_update_time,
            reward_fn=self.reward_fn,
            sumo=conn
        )
        conn.close()

    def step(self, action):
        # Ensure action is within the valid range
        action = max(0, min(action, self.traffic_signal.num_green_phases - 1))
        do_action = self.traffic_signal.update(action)
        self.sumo.simulationStep()
        next_state = self.compute_state()
        reward = self._compute_reward(self.train_state, action)
        done = self._compute_done()
        self._compute_average_queue(done)
        info = {'do_action': do_action}
        return next_state, reward, done, info

    def _random_skip(self):
        rand_idx = np.random.randint(0, self.skip_range)
        next_state = None  # Initialize next_state
        for _ in range(rand_idx):
            next_state, _, _, _ = self.step(rand_idx)
        return next_state

    def reset(self):
        try:
            traci.close()
        except traci.FatalTraCIError:
            pass
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        self.sumo = traci
        self.traffic_signal.reset()
        self.train_state = self.compute_state()
        self.next_state = None
        self.last_phase_state = None
        self.change_action_time = None
        self.queue = []
        self.avg_queue = []
        self.avg_wait = []
        self.total_rewards = 0
        return self._random_skip()

    def render(self):
        pass

    def close(self):
        self.sumo.close()

    def seed(self, seed=None):
        pass

    # state -> real time state; train_state -> internal state for train
    def compute_state(self):
        return self.traffic_signal.compute_state()

    def _compute_reward(self, state, action):
        ts_reward = self.traffic_signal.compute_reward(state, action)
        return ts_reward

    def _compute_done(self):
        current_time = self.sumo.simulation.getTime()
        return current_time > self.simulation_time

    def _compute_average_queue(self, done):
        if done and len(self.queue) > 0:
            self.avg_queue.append(np.mean(self.queue))
            self.queue = []
        q = self.traffic_signal.compute_queue()
        self.queue.append(q)

    def _compute_average_wait(self, done):
        if done and len(self.avg_wait) > 0:
            self.avg_wait.append(np.mean(self.avg_wait))
            self.avg_wait = []
        waiting_time = self.traffic_signal.compute_waiting_time()
        self.avg_wait.append(waiting_time)

    def _compute_total_rewards(self, reward, done):
        if reward is None:
            reward = 0

        self.total_rewards += reward

    @property
    def observation_space(self):
        return self.traffic_signal.observation_space

    @property
    def action_space(self):
        return self.traffic_signal.action_space
