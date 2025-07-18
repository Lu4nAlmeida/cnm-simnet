import numpy as np
import matplotlib.pyplot as plt
import PMmodels_benchmark_tests as pm

# Initial Parameters
num_dim = 2
time_step = 0.01
duration = 1.0

true_times = []
pred_times = []
accuracies = []

for i in range(1000):
    expected, predicted, *results = pm.simulation_comparison(num_dim, time_step, duration)
    true_times.append(results[0])
    pred_times.append(results[1])
    accuracies.append(results[2])

pm.print_results(np.mean(np.array(true_times)), np.mean(np.array(pred_times)), np.mean(np.array(accuracies)))