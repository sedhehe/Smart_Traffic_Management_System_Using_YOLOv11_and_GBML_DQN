import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_output(outputs):
    with torch.no_grad():
        for m, output in enumerate(outputs):
            if output is None:
                continue
            max_t = torch.max(torch.abs(output))
            if abs(max_t - 0) < 1e-2:
                continue
            for n, t in enumerate(output):
                if t is None:
                    continue
                outputs[m][n] /= max_t
    return outputs

def compute_waiting_time(self):
    """
    Computes the total waiting time of all vehicles in the controlled lanes.
    :return: Total waiting time of vehicles in all controlled lanes.
    """
    total_waiting_time = 0
    for lane_id in self.lanes_id:
        vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
        for vehicle_id in vehicles:
            total_waiting_time += self.sumo.vehicle.getWaitingTime(vehicle_id)
    return total_waiting_time

class DqnNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(inputs, 512)
        self.l2 = nn.Linear(512, outputs)
        c = np.sqrt(1 / inputs)
        nn.init.uniform_(self.l1.weight, -c, c)
        nn.init.uniform_(self.l1.bias, -c, c)
        nn.init.uniform_(self.l2.weight, -c, c)
        nn.init.uniform_(self.l2.bias, -c, c)

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.l1(x))
        x = self.l2(x)
        return x
