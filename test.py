import numpy as np
import matplotlib.pyplot as plt
import sympy
import math
import time

def velocity_derivative(acc, mass, drg, t):
    return acc * math.e ** (-(drg * t) / mass)

def displacement(acc, mass, drg, t, C=0):
    return (mass * acc / drg) * t - (mass**2 * acc / drg**2) * math.e ** (-(drg * t) / mass) + C

# Example usage:
acc = 9.8  # Example acceleration (e.g., gravity)
mass = 10  # Example mass
drg = 2   # Example drag coefficient
t = 5      # Example time

# Measure start time
start_time = time.perf_counter()

print(velocity_derivative(acc, mass, drg, t))
print(displacement(acc, mass, drg, t))

# Measure end time
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Step computed in {elapsed_time:.9f} seconds.")


random_array = np.random.uniform(0, 10, 100)
#print(random_array)


def matrix_sort(input_data):
    a = 10
    if input_data[0] > input_data[1]:
        b = input_data[0]
        input_data[0] = input_data[1]
        input_data[1] = b
    if input_data[2] < input_data[3]:
        b = input_data[2]
        input_data[2] = input_data[3]
        input_data[3] = b
    if input_data[0] > input_data[3]:
        b = input_data[0]
        input_data[0] = input_data[3]
        input_data[3] = b
    return input_data
