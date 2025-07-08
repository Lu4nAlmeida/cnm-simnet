import numpy as np
import pickle as pk
import time

i = time.time()

# Parameters
num_dim = 2
time_step = 0.05
duration = 5
simulations = 100

input_data = []
output_data = []

for simulation in range(simulations):
    # Initial conditions
    initial = {"pos": np.random.uniform(-3, 3, num_dim),
               "vel": np.random.uniform(-3, 3, num_dim)}
    acc = np.random.uniform(-3, 3, num_dim)

    # Time steps as a NumPy array
    delta_times = np.arange(0, duration, time_step)

    # Compute displacement and velocity for all delta_times
    final_pos = initial["pos"] + initial["vel"] * delta_times[:, None] + 0.5 * acc * (delta_times[:, None] ** 2)
    final_vel = initial["vel"] + acc * delta_times[:, None]

    # Prepare inputs: pos, vel, acc, delta_time
    inputs = np.column_stack([
        np.tile(initial["pos"], (len(delta_times), 1)),  # Initial position (repeated)
        np.tile(initial["vel"], (len(delta_times), 1)),  # Initial velocity (repeated)
        np.tile(acc, (len(delta_times), 1)),            # Acceleration (repeated)
        delta_times[:, None]                            # Delta times
    ])

    # Prepare outputs: pos, vel
    outputs = np.hstack([final_pos, final_vel])  # Final positions and velocities

    input_data.append(inputs)
    output_data.append(outputs)

input_data = np.array(input_data)
output_data = np.array(output_data)

f = time.time()

with open(r"C:\Users\LUAN\PycharmProjects\NeuralNetwork\inputs.pkl", 'wb') as file:
    pk.dump(input_data, file)

with open(r"C:\Users\LUAN\PycharmProjects\NeuralNetwork\outputs.pkl", 'wb') as file:
    pk.dump(output_data, file)

print(f"It took {f-i:.3f} seconds to generate all data.")
