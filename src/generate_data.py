import torch
import numpy as np
import time
from simulate_ldpm import ProjectileMotionSimulation
from simulate_kinematics import KinematicsSimulation
import copy

def generate_ldpm_data(X_scaler, y_scaler, num_simulations, num_dim, shape, time_steps):
    input_data = []
    output_data = []
    runtimes = []

    for _ in range(num_simulations):
        ldpm = ProjectileMotionSimulation()
        ldpm.random_initial_parameters(num_dim, shape)

        start_time = time.perf_counter()
        ldpm.compute_output(time_steps)
        end_time = time.perf_counter()

        sim_time = end_time - start_time
        runtimes.append(sim_time)

        inputs = np.concatenate([
            ldpm.vel,
            ldpm.acc,
            ldpm.mass[:, :-1],
            ldpm.drg[:, :-1],
        ], axis=1)
        outputs = np.concatenate([
            ldpm.pos,
            ldpm.vel,
            ldpm.frc
        ], axis=1)

        input_data.append(inputs)
        output_data.append(outputs)

    runtimes = np.array(runtimes)
    sim_time = np.mean(runtimes)
    total_time = np.sum(runtimes)

    X = np.array(input_data).reshape(-1, num_dim * 2 + 2)
    y = np.array(output_data).reshape(-1, num_dim * 3)

    X_norm = X_scaler.transform(X).reshape(num_simulations, shape[0], num_dim * 2 + 2)
    y_norm = y_scaler.transform(y).reshape(num_simulations, shape[0], num_dim * 3)

    # Train/test split
    split = int(0.8 * len(X_norm))
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y_norm[:split], y_norm[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, sim_time, total_time


def generate_kinematics_data(num_simulations, num_dim, shape, time_steps):
    input_data = []
    output_data = []
    runtimes = []

    for _ in range(num_simulations):
        initial = KinematicsSimulation()
        initial.random_initial_parameters(num_dim, shape)
        final = copy.deepcopy(initial)

        start_time = time.perf_counter()
        final.compute_output(time_steps)
        end_time = time.perf_counter()

        sim_time = end_time - start_time
        runtimes.append(sim_time)

        inputs = np.concatenate([
            initial.vel,
            initial.acc,
            time_steps
        ], axis=1)
        outputs = np.concatenate([
            final.pos,
            final.vel,
        ], axis=1)

        input_data.append(inputs)
        output_data.append(outputs)

    runtimes = np.array(runtimes)
    sim_time = np.mean(runtimes)
    total_time = np.sum(runtimes)

    # Shuffle data
    perm = np.random.permutation(X.shape[0])
    X_shuffled = np.array(input_data)[perm]
    y_shuffled = np.array(output_data)[perm]

    X_shuffled = X_shuffled.reshape(-1, num_dim * 2 + 1)
    y_shuffled = y_shuffled.reshape(-1, num_dim * 2)

    # Train/test split
    split = int(0.8 * len(X_shuffled))
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, sim_time, total_time
