import numpy as np
import pickle as pk
import time
import math
import random

i = time.time()

# Parameters
num_dim = 2
time_step = 0.01
t = 1.3
shape = (int(t/time_step), num_dim)
simulations = 100

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

for simulation in range(simulations):
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

    # Prepare outputs: pos, vel, frc
    outputs = np.concatenate([
        final["pos"],
        final["vel"],
        final["frc"],
    ], axis=1)

    input_data.append(inputs)
    output_data.append(outputs)

input_data = np.array(input_data)
output_data = np.array(output_data)

f = time.time()

with open(r"C:\Users\LUAN\PycharmProjects\NeuralNetwork\inputs.pkl", 'wb') as file:
    pk.dump(input_data, file)

with open(r"C:\Users\LUAN\PycharmProjects\NeuralNetwork\outputs.pkl", 'wb') as file:
    pk.dump(output_data, file)

print(f"It took {f-i:.3f} seconds to generate all data.")  # It took 0.019 seconds to generate all data.
