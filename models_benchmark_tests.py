import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import profile


class KinematicsModel(nn.Module):
    def __init__(self, input_features=7, output_features=4, hidden_units=256, activation=nn.ReLU()):
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

@profile
def simulation_comparison():
    # Parameters
    num_dim = 2
    time_step = 0.05
    t = 1
    input_data = []

    # Initial conditions
    initial = {"pos": np.array([0,0]),
               "vel": np.random.uniform(-3, 3, num_dim)}

    # acc = np.random.uniform(-3, 3, num_dim)
    acc = np.array([0, -2])

    # Time steps as a NumPy array
    delta_times = np.arange(0, t, time_step)

    # Prepare inputs: pos, vel, acc, delta_time
    inputs = np.column_stack([
        np.tile(np.array([0, 0]), (len(delta_times), 1)),  # Initial position (repeated)
        np.tile(initial["vel"], (len(delta_times), 1)),  # Initial velocity (repeated)
        np.tile(acc, (len(delta_times), 1)),            # Acceleration (repeated)
        delta_times[:, None]                            # Delta times
    ])
    input_data.append(inputs)
    input_data = torch.tensor(np.array(input_data), dtype=torch.float32)

    # Measure start time
    start_time = time.perf_counter()

    # Compute displacement and velocity for all delta_times
    final = {"pos": initial["vel"] * delta_times[:, None] + 0.5 * acc * (delta_times[:, None] ** 2),
             "vel": initial["vel"] + acc * delta_times[:, None]}

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Simulation computed in {elapsed_time:.9f} seconds.")

    model = KinematicsModel()
    model.load_state_dict(torch.load("/Users/luanalmeidatobias/PycharmProjects/Polygence-Research/Kinematics Models/model_8.pth", map_location=torch.device('cpu')))
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

    # True Simulation Visualization
    for i, delta_time in enumerate(delta_times):
        pos = final["pos"][i]
        vel = final["vel"][i]

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Plot velocity and acceleration vectors
        #plt.quiver(pos[0], pos[1], acc[0], acc[1], color='red', scale=15)  # Acceleration
        #plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

        # Plot particle position
        plt.scatter(pos[0], pos[1], color="black")


    # Inferred Simulation Visualization
    for i, delta_time in enumerate(delta_times):
        pos = inferred_pos[0][i] + initial["pos"]
        vel = inferred_vel[0][i]

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Plot velocity and acceleration vectors
        #plt.quiver(pos[0], pos[1], acc[0], acc[1], color='red', scale=15)  # Acceleration
        #plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

        # Plot particle position
        plt.scatter(pos[0], pos[1], color="green")

    plt.show()

simulation_comparison()