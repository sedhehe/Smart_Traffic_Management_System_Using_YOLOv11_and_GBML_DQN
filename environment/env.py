import gym
import os
import sys
import random
from environment.traffic_signal import TrafficSignal
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
import sumolib

class SumoEnv(gym.Env):
    def __init__(self, net_file, route_file, skip_range, simulation_time, yellow_time, delta_rs_update_time, reward_fn, mode, use_gui=False):
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
        self.queue = []
        self.avg_queue = []

        self.sumoBinary = 'sumo-gui' if self.use_gui else 'sumo'

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        conn = traci
        self.ts_id = traci.trafficlight.getIDList()[0]
        self.traffic_signal = TrafficSignal(ts_id=self.ts_id,
                                            yellow_time=self.yellow_time,
                                            simulation_time=simulation_time,
                                            delta_rs_update_time=delta_rs_update_time,
                                            reward_fn=self.reward_fn,
                                            sumo=conn)
        conn.close()

    def step(self, action):
        if self.traffic_signal.handle_emergency_vehicle():
            self.sumo.simulationStep()
            return self.compute_state(), None, False, {'do_action': -1}

        next_state = None
        reward = None
        start = False
        done = False
        info = {'do_action': None}
        do_action = self.traffic_signal.change_phase(action)
        if do_action is None:
            return next_state, reward, done, info

        self.sumo.simulationStep()
        if do_action == -1 and self.change_action_time is None:
            self.change_action_time = self.sumo.simulation.getTime() + self.yellow_time

        if self.change_action_time is not None and self.sumo.simulation.getTime() >= self.change_action_time:
            self.change_action_time = None
            self.train_state = self.compute_state()
            start = True

        next_state = self._compute_next_state()
        reward = self._compute_reward(start, do_action)
        done = self._compute_done()
        info = {'do_action': do_action}
        self._compute_average_queue(done)
        return next_state, reward, done, info

    def _random_skip(self):
        self.traffic_signal.sumo = self.sumo
        rand_idx = random.randint(0, len(self.traffic_signal.all_green_phases)-1)
        self.traffic_signal.yellow_phase = None
        self.traffic_signal.green_phase = self.traffic_signal.all_green_phases[rand_idx]
        self.sumo.trafficlight.setRedYellowGreenState(self.traffic_signal.ts_id, self.traffic_signal.green_phase.state)
        self.traffic_signal.update_end_time()
        self.traffic_signal.rs_update_time = 0

        skip_seconds = random.randint(0, self.skip_range)
        initial_state = self.compute_state()
        if self.mode == 'train':
            for _ in range(skip_seconds):
                rand_idx = random.randint(0, len(self.traffic_signal.all_green_phases)-1)
                next_state, _, _, _ = self.step(rand_idx)
                if next_state is not None:
                    initial_state = next_state

        return initial_state

    def reset(self):
        sumo_cmd = [sumolib.checkBinary(self.sumoBinary), '-n', self._net, '-r', self._route, '--time-to-teleport', '1000']
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        traci.start(sumo_cmd)
        self.sumo = traci
        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        return self._random_skip()

    def render(self):
        pass

    def close(self):
        self.sumo.close()

    def seed(self, seed=None):
        pass

    def compute_state(self):
        return self.traffic_signal.compute_state()

    def _compute_next_state(self):
        next_state = self.traffic_signal.compute_next_state()
        if next_state is not None:
            self.next_state = next_state
        return next_state

    def _compute_reward(self, start, do_action):
        return self.traffic_signal.compute_reward(start, do_action)

    def _compute_done(self):
        return self.sumo.simulation.getTime() > self.simulation_time

    def _compute_average_queue(self, done):
        if done and self.queue:
            self.avg_queue.append(np.mean(self.queue))
            self.queue = []
        self.queue.append(self.traffic_signal.compute_queue())

    def compute_average_waiting_time(self):
        total_waiting_time = 0
        total_vehicles = 0
        for lane_id in self.traffic_signal.lanes_id:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in veh_list:
                total_waiting_time += self.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
                total_vehicles += 1
        return total_waiting_time / total_vehicles if total_vehicles else 0

    def compute_total_reward(self):
        total_reward = 0
        for lane_id in self.traffic_signal.lanes_id:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in veh_list:
                total_reward += self.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
        return -total_reward

    @property
    def observation_space(self):
        return self.traffic_signal.observation_space

    @property
    def action_space(self):
        return self.traffic_signal.action_space
