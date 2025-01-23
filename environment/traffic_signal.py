import numpy as np
from gym import spaces

class TrafficSignal:
    def __init__(
        self,
        ts_id: str,
        yellow_time: int,
        simulation_time: float,
        delta_rs_update_time: int,
        reward_fn: str,
        sumo
    ):
        self.ts_id = ts_id
        self.yellow_time = yellow_time
        self.simulation_time = simulation_time
        self.delta_rs_update_time = delta_rs_update_time
        # reward_state_update_time
        self.rs_update_time = 0
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.green_phase = None
        self.yellow_phase = None
        self.end_min_time = 0
        self.end_max_time = 0
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        self.num_green_phases = len(self.all_green_phases)
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(self.num_green_phases)
        self.last_measure = 0
        self.dict_lane_veh = None

    def reset(self):
        self.green_phase = None
        self.yellow_phase = None
        self.end_min_time = 0
        self.end_max_time = 0
        self.rs_update_time = 0
        self.last_measure = 0
        self.dict_lane_veh = None

    def update(self, action):
        """
        Update the traffic signal phase based on the action.
        :param action: The action to be taken (new green phase index).
        """
        # Ensure action is within the valid range
        if action < 0 or action >= self.num_green_phases:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.num_green_phases - 1}.")
        
        do_action = self.change_phase(action)
        return do_action

    def change_phase(self, new_green_phase):
        """
        :param new_green_phase:
        :return: do_action -> the real action operated; if is None, means the new_green_phase is not appropriate,
        need to choose another green_phase and operate again
        """
        # yellow_phase has not finished yet
        # yellow_phase only has duration, no minDur or maxDur
        if self.is_emergency_present():
            print("Emergency vehicle detected! Adjusting signals.")
            emergency_lane = self._check_emergency_vehicle()
            self._handle_emergency(emergency_lane)
            return -1

        # do_action mapping (int -> Phase)
        new_green_phase = self.all_green_phases[new_green_phase]
        do_action = new_green_phase
        current_time = self.sumo.simulation.getTime()
        
        # Initialize green_phase if None
        if self.green_phase is None:
            self.green_phase = new_green_phase

        if self.yellow_phase is not None:
            if current_time >= self.end_max_time:
                self.yellow_phase = None
                self.update_end_time()
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                do_action = self.green_phase
            else:
                do_action = self.yellow_phase
        else:
            # if old_green_phase has finished
            if current_time >= self.end_min_time:
                if new_green_phase.state == self.green_phase.state:
                    if current_time < self.end_max_time:
                        do_action = self.green_phase
                    else:
                        # current phase has reached the max operation time, have to find another green_phase instead
                        do_action = None
                else:
                    # need to set a new plan(yellow + new_green)
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    self.green_phase = new_green_phase
                    self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time
                    self.update_end_time()
                    do_action = self.yellow_phase
            else:
                do_action = self.green_phase

        if do_action is None:
            return None

        # Ensure do_action is an integer
        if isinstance(do_action, int):
            return do_action

        if 'y' in do_action.state:  # Yellow phase
            return -1

        # Find the corresponding action index for the green phase
        for i, green_phase in enumerate(self.all_green_phases):
            if do_action.state == green_phase.state:
                return i

        # If no match is found, return None
        return None

    
    def is_emergency_present(self):
        """
        Checks if any emergency vehicle is present in the controlled lanes.
        :return: True if an emergency vehicle is detected, otherwise False.
        """
        for lane_id in self.lanes_id:
            vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles:
                if self.sumo.vehicle.getTypeID(vehicle_id) == "emergency":
                    return True
        return False
    
    def _check_emergency_vehicle(self):
        for lane_id in self.lanes_id:
            vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles:
                if self.sumo.vehicle.getTypeID(vehicle_id) == "emergency":
                    return lane_id
        return None
    
    def _handle_emergency(self, emergency_lane):
        if emergency_lane is None:
            return  # No action if no emergency lane is provided
        print(f"Handling emergency vehicle in lane: {emergency_lane}")
        emergency_phase_state = ""
        for lane_id in self.lanes_id:
            if lane_id == emergency_lane:
                emergency_phase_state += "G"  # Green for the emergency lane
            else:
                emergency_phase_state += "r"  # Red for all other lanes

        # Create and set the emergency phase
        emergency_phase = self.sumo.trafficlight.Phase(30, emergency_phase_state)
        self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, emergency_phase.state)
        self.green_phase = emergency_phase
        self.yellow_phase = None
        self.update_end_time()
        print("Emergency phase applied successfully.")

    def update_end_time(self):
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is None:
            self.end_min_time = current_time + self.green_phase.minDur
            self.end_max_time = current_time + self.green_phase.maxDur
        else:
            self.end_min_time = current_time + self.yellow_time
            self.end_max_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        update_reward = False
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            update_reward = True
        if self.reward_fn == 'choose-min-waiting-time':
            return self._choose_min_waiting_time(start, update_reward, do_action)
        else:
            return None

    def _choose_min_waiting_time(self, start, update_reward, do_action):
        if np.any(start):  # Use np.any() to check if any element in start is True
            self.dict_lane_veh = {}
            for lane_id in self.lanes_id:
                veh_list = self.sumo.lane.getLastStepVehicleIDs(lane_id)
                wait_veh_list = [veh_id for veh_id in veh_list if self.sumo.vehicle.getAccumulatedWaitingTime(veh_id) > 0]
                self.dict_lane_veh[lane_id] = len(wait_veh_list)

            # Dynamically map actions to lane groups
            lane_groups = [
                [lane_id for lane_id in self.lanes_id if 'n_t' in lane_id or 's_t' in lane_id],
                [lane_id for lane_id in self.lanes_id if 'e_t' in lane_id or 'w_t' in lane_id]
            ]

            rewards = []
            for group in lane_groups:
                group_waiting = sum(self.dict_lane_veh[lane_id] for lane_id in group)
                rewards.append(-group_waiting)

            if update_reward:
                self.last_measure = rewards

            return rewards[do_action] if do_action < len(rewards) else None
        return None
