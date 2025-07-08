import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import copy
from memory_profiler import profile


class ProjectileMotionModel(nn.Module):
    def __init__(self, input_features=7, output_features=6, hidden_units=256, activation=nn.ReLU()):
        '''
        Args:
          input_features: number of input neurons (num_dim * 2 + 3)
          output_features: number of output neurons (num_dim * 3)
          hidden_units: number of neurons per hidden layer

        Returns:
          A PyTorch model that predicts the motion of a particle based on set initial conditions.
        '''
        super().__init__()
        self.linear_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=hidden_units), # Input Layer
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units), # Hidden Layer 1
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units), # Hidden Layer 2
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=output_features) # Output Layer
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class ProjectileMotionSimulation:
    def __init__(self, pos, vel, mass, acc, drg):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.mass = mass
        self.acc = np.array(acc)
        self.drg = drg
        self.frc = self.mass * self.acc

    def compute_output(self, initial, delta_times):
        self.pos = self.delta_position(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)
        self.vel = self.final_velocity(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)
        self.frc = self.drag_force(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)

    @staticmethod
    def final_velocity(mass, acc, drg, vel, t):
        return (vel - (mass * acc) / drg) * math.e ** (-(drg * t) / mass) + (mass * acc) / drg

    @staticmethod
    def delta_position(mass, acc, drg, vel, t):
        return (mass / drg) * (vel - (mass * acc) / drg) * (1 - math.e ** ((-drg * t) / mass)) + (mass * acc * t) / drg

    @staticmethod
    def drag_force(mass, acc, drg, vel, t=0):
        return -(drg * ProjectileMotionSimulation.final_velocity(mass, acc, drg, vel, t))


def simulation_visualization(components, frames, color="black"):
    for i in range(frames):
        if i % 10 == 0:
            pos = components[0][i]
            vel = components[1][i]
            frc = components[2][i]

            # Set axis limits dynamically
            plt.xlim(-9, 9)
            plt.ylim(-9, 9)

            # Plot velocity and acceleration vectors
            # plt.quiver(pos[0], pos[1], frc[0], frc[1], color='orange', scale=15)  # Drag Force
            # plt.quiver(pos[0], pos[1], initial["acc"][0][0], initial["acc"][0][1], color='red', scale=15)  # Acceleration
            # plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

            # Plot particle position
            plt.scatter(pos[0], pos[1], color=color)


def accuracy(expected_output, inferred_output):
    error = (expected_output - inferred_output)**2
    accuracy_matrix = 1 / (error + 1)
    return np.mean(accuracy_matrix) * 100

@profile
def simulation_comparison(num_dim, time_step, duration):
    shape = (int(duration / time_step), num_dim)

    # Time steps as a NumPy array
    delta_times = np.arange(0, duration, time_step)
    delta_times = np.broadcast_to(delta_times[:, None], shape)  # Shape: (130, 2)

    # Initial conditions
    initial = ProjectileMotionSimulation(
        pos=np.broadcast_to(np.zeros(num_dim), shape),
        vel=np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
        mass=np.full(shape, random.uniform(0, 10)),
        acc=np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
        drg=np.full(shape, random.uniform(0, 2))
    )

    # Final conditions
    final = copy.deepcopy(initial)

    start_time = time.perf_counter()
    final.compute_output(initial, delta_times)
    end_time = time.perf_counter()

    true_time = end_time - start_time


    # Prepare inputs: vel, acc, mass, drg, delta_times
    input_data = np.concatenate([
        initial.vel,
        initial.acc,
        initial.mass[:, :-1],
        initial.drg[:, :-1],
        delta_times[:, :-1]  # Delta times
    ], axis=1)

    model = ProjectileMotionModel()
    model.load_state_dict(torch.load("/Users/luanalmeidatobias/PycharmProjects/Polygence-Research/Projectile Motion Models/PMmodel_0.pth", map_location=torch.device('cpu')))
    model.to(device='cpu')
    input_data = torch.tensor(input_data).to(device="cpu").float()
    model.eval()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_data)

    end_time = time.perf_counter()
    pred_time = end_time - start_time

    inferred_pos = output[:, :2].cpu().detach().numpy()
    inferred_vel = output[:, 2:4].cpu().detach().numpy()
    inferred_frc = output[:, 4:].cpu().detach().numpy()

    prediction_accuracy = accuracy(
                    expected_output=np.array([final.pos, final.vel, final.frc]),
                    inferred_output=np.array([inferred_pos, inferred_vel, inferred_frc])
                    )

    return (final.pos, final.vel, final.frc), (inferred_pos, inferred_vel, inferred_frc), true_time, pred_time, prediction_accuracy

def print_results(true_time, pred_time, prediction_accuracy):
    print(f"Simulation computed in {true_time:.9f} seconds.")
    print(f"Simulation inferred in {pred_time:.9f} seconds.")
    print(f"Accuracy: {prediction_accuracy:.2f}%")
    print(f"Computational time ratio: {true_time / pred_time:.3f}")

simulation_comparison(2, 0.01, 1.3)