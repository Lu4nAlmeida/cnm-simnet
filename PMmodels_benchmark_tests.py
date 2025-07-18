import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import copy
import pickle as pk
from models import *


def predict_physics(input_data):
    # Load the input and output scalers
    with open("/Users/luanalmeidatobias/PycharmProjects/Polygence-Research/Projectile Motion Models/scalers5.pkl", "rb") as f:
        X_scaler, y_scaler = pk.load(f)

    # Normalize input_data and convert to Tensor
    input_data_normalized = X_scaler.transform(input_data)
    input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)

    # Reshape to (batch, seq_len, input_dim) → (1, seq_len, 6)
    input_tensor = input_tensor.unsqueeze(0)

    # Load model
    model = ProjectileMotionModelRNN()
    model.load_state_dict(torch.load("/Users/luanalmeidatobias/PycharmProjects/Polygence-Research/Projectile Motion Models/PMmodel_5.pth", map_location=torch.device('cpu')))
    model.to(device='cpu')
    model.eval()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(input_tensor)  # output shape: (1, seq_len, 6)

    end_time = time.perf_counter()
    pred_time = end_time - start_time

    # Remove batch dimension and unnormalize
    output = output.squeeze(0).cpu().numpy()
    output_real = y_scaler.inverse_transform(output)

    # Return pos, vel, frc
    return [output_real[:, :2], output_real[:, 2:4], output_real[:, 4:]], pred_time


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
            #plt.quiver(pos[0], pos[1], frc[0], frc[1], color='orange', scale=15)  # Drag Force
            #plt.quiver(pos[0], pos[1], initial["acc"][0][0], initial["acc"][0][1], color='red', scale=15)  # Acceleration
            #plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

            # Plot particle position
            plt.scatter(pos[0], pos[1], color=color)


def accuracy(expected_output, inferred_output):
    error = (expected_output - inferred_output)**2
    accuracy_matrix = 1 / (error + 1)
    return np.mean(accuracy_matrix) * 100


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

    final.compute_output(initial, delta_times) # True computation (expected output)

    end_time = time.perf_counter()
    true_time = end_time - start_time


    # Prepare inputs: vel, acc, mass, drg, delta_times
    input_data = np.concatenate([
        initial.vel,
        initial.acc,
        initial.mass[:, :-1],
        initial.drg[:, :-1],
    ], axis=1)

    output, pred_time = predict_physics(input_data)

    prediction_accuracy = accuracy(
        expected_output=np.array([final.pos, final.vel, final.frc]),
        inferred_output=np.array(output)
    )

    return (final.pos, final.vel, final.frc), output, true_time, pred_time, prediction_accuracy


def print_results(true_time, pred_time, prediction_accuracy):
    print(f"Simulation computed in {true_time:.9f} seconds.")
    print(f"Simulation inferred in {pred_time:.9f} seconds.")
    print(f"Accuracy: {prediction_accuracy:.2f}%")
    print(f"Computational time ratio: {true_time / pred_time:.3f}")


true, pred, *_ = simulation_comparison(2, 0.01, 1.0)
simulation_visualization(true, 100)
simulation_visualization(pred, 100, color="blue")
plt.show()