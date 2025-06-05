import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math


class ProjectileMotionModel(nn.Module):
    def __init__(self, input_features=7, output_features=6, hidden_units=256, activation=nn.ReLU()):
        '''
        Args:
          input_features: number of input neurons (num_dim * 3 + 1)
          output_features: number of output neurons (num_dim * 2)
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


# Parameters
num_dim = 2
time_step = 0.01
t = 1.3
shape = (int(t/time_step), num_dim)

input_data = []
output_data = []


def drag_force(mass, acc, drg, vel, t=0):
    return -(drg * final_velocity(mass, acc, drg, vel, t))


def final_velocity(mass, acc, drg, vel, t):
    return (vel - (mass * acc)/drg) * math.e**(-(drg * t)/mass) + (mass * acc)/drg


def delta_position(mass, acc, drg, vel, t):
    return (mass/drg) * (vel - (mass * acc)/drg) * (1 - math.e**((-drg * t) / mass)) + (mass * acc * t) / drg


# Time steps as a NumPy array
delta_times = np.arange(0, t, time_step)
delta_times = np.broadcast_to(delta_times[:, None], shape)  # Shape: (130, 2)


# Initial conditions
initial = {"pos": np.broadcast_to(np.zeros(num_dim), shape),
           "vel": np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
           "mass": np.full(shape, random.uniform(0, 10)),
           "acc": np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
           "drg": np.full(shape, random.uniform(0, 2))}

# Final conditions
final = {
    "pos": delta_position(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times),
    "vel": final_velocity(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times),
    "frc": drag_force(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times)}

# Prepare inputs: vel, acc, mass, drg, delta_times
inputs = np.concatenate([
    initial["vel"],
    initial["acc"],
    initial["mass"][:, :-1],
    initial["drg"][:, :-1],
    delta_times[:, :-1]  # Delta times
], axis=1)

model = ProjectileMotionModel()
model.load_state_dict(torch.load("C:\\Users\\LUAN\\PycharmProjects\\NeuralNetwork\\Projectile Motion Models\\PMmodel_0.pth", map_location=torch.device('cpu')))
model.to(device='cpu')
input_data = input_data.to(device='cpu')
model.eval()

start_time = time.perf_counter()

with torch.no_grad():
    output = model(input_data)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Simulation inferred in {elapsed_time:.9f} seconds.")

inferred_pos = output[:, :, :2].cpu().detach().numpy()
inferred_vel = output[:, :, 2:4].cpu().detach().numpy()
inferred_frc = output[:, :, 2:4].cpu().detach().numpy()

# True Simulation Visualization
for i, delta_time in enumerate(delta_times):
    plt.clf()

    pos = final["pos"][i]
    vel = final["vel"][i]

    # Set axis limits dynamically
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Plot velocity and acceleration vectors
    plt.quiver(pos[0], pos[1], acc[0], acc[1], color='red', scale=15)  # Acceleration
    plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

    # Plot particle position
    plt.scatter(pos[0], pos[1], color="black")
    plt.pause(0.0001)

plt.show()

# Inferred Simulation Visualization
for i, delta_time in enumerate(delta_times):
    plt.clf()

    pos = inferred_pos[0][i]
    vel = inferred_vel[0][i]

    # Set axis limits dynamically
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Plot velocity and acceleration vectors
    plt.quiver(pos[0], pos[1], acc[0], acc[1], color='red', scale=15)  # Acceleration
    plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

    # Plot particle position
    plt.scatter(pos[0], pos[1], color="black")
    plt.pause(0.0001)

plt.show()
