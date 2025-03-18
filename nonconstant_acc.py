import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class KinematicsModel(nn.Module):
    def __init__(self, input_features=7, output_features=4, hidden_units=256, activation=nn.ReLU()):
        """
        Args:
          input_features: number of input neurons (num_dim * 3 + 1)
          output_features: number of output neurons (num_dim * 2)
          hidden_units: number of neurons per hidden layer

        Returns:
          A PyTorch model that predicts the motion of a particle based on set initial conditions.
        """
        super().__init__()
        self.linear_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=hidden_units),  # Input Layer
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units),  # Hidden Layer 1
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units),  # Hidden Layer 2
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=output_features)  # Output Layer
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


def acceleration(x):
    return 2 * np.sin(5*x)  # 5**x | 2 * np.sin(5*x)


# Parameters
num_dim = 2
time_step = 0.01
t = 1
input_data = []

# Initial conditions
initial = {"pos": np.random.uniform(-3, 3, num_dim),
           "vel": np.random.uniform(-3, 3, num_dim)}

# Time steps as a NumPy array
delta_times = np.arange(0, t, time_step)

# Non-constant acceleration
acc = []
for frame in delta_times:
    acc.append([acceleration(frame), 0])  # Use numpy-based acceleration
acc = np.array(acc, dtype=np.float32)  # Ensure the array is of type float32

# Prepare inputs: pos, vel, acc, delta_time
inputs = np.column_stack([
    np.tile(initial["pos"], (len(delta_times), 1)),  # Initial position (repeated)
    np.tile(initial["vel"], (len(delta_times), 1)),  # Initial velocity (repeated)
    acc,                                             # Acceleration
    delta_times[:, None]                             # Delta times
])
input_data.append(inputs)
input_data = torch.tensor(np.array(input_data, dtype=np.float32), dtype=torch.float32)  # Ensure numeric type

# Measure start time
start_time = time.perf_counter()

# Compute velocity and position using numpy
acc_values = np.array([acceleration(frame) for frame in delta_times], dtype=np.float32)

# Integrate acceleration to get velocity (using cumulative sum for numerical integration)
delta_vel = np.cumsum(acc_values) * time_step  # Approximate integral of acceleration
vel = initial["vel"] + delta_vel[:, None]  # Add initial velocity

# Integrate velocity to get position (using cumulative sum for numerical integration)
delta_pos = np.cumsum(vel, axis=0) * time_step  # Approximate integral of velocity
final = {"pos": initial["pos"] + delta_pos,
         "vel": vel}

# Measure end time
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Simulation computed in {elapsed_time:.9f} seconds.")

model = KinematicsModel()
model.load_state_dict(torch.load("C:\\Users\\LUAN\\PycharmProjects\\NeuralNetwork\\model_8.pth", map_location=torch.device('cpu')))
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
    plt.clf()

    pos = final["pos"][i]
    vel = final["vel"][i]

    # Set axis limits dynamically
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Plot velocity and acceleration vectors
    plt.quiver(pos[0], pos[1], acc[i, 0], acc[i, 1], color='red', scale=15)  # Acceleration
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
    plt.quiver(pos[0], pos[1], acc[i, 0], acc[i, 1], color='red', scale=15)  # Acceleration
    plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

    # Plot particle position
    plt.scatter(pos[0], pos[1], color="black")
    plt.pause(0.0001)

plt.show()