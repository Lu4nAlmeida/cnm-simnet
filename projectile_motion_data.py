import matplotlib.pyplot as plt
import numpy as np
import sympy
import math
import time

'''
WARNING
When I started coding this, only God and I knew what was going on...
Now only God knows.
'''

num_dim = 2
time_step = 0.01
total_time = 1

# Initial conditions
initial = {"pos": np.zeros(num_dim),
           "vel": np.random.uniform(-3, 3, num_dim),
           "mass": 1.0,
           "acc": np.random.uniform(-3, 3, num_dim),
           "drg": 1.0}

# Final conditions
final = {"pos": initial["pos"],
         "vel": initial["vel"],
         "frc": np.zeros(num_dim)}


def force(mass, acc, drg, vel_0):
    return (mass * acc) - (vel_0 * drg)  # F = ma | Fnet = sum(F) | Fd = mdv/dt - bv


def velocity(acc, mass, drg, t, vel):
    return ((mass * acc - mass * acc * math.e**(-(drg * t) / mass)) / drg) + vel


def displacement(acc, mass, drg, t, pos):  # take integral of velocity() with respect to t
    return (mass * acc / drg) * t - (mass**2 * acc / drg**2) * math.e ** (-(drg * t) / mass) + pos


loop_time = time.time()
# Main loop
for i in range(int(total_time / time_step)):
    plt.clf()
    delta_time = i * time_step  # Elapsed time

    # Measure start time
    start_time = time.perf_counter()
    for dim in range(num_dim):
        final["pos"][dim] = displacement(initial["acc"][dim], initial["mass"], initial["drg"], delta_time, final["pos"][dim])
        final["vel"][dim] = velocity(initial["acc"][dim], initial["mass"], initial["drg"], delta_time, final["vel"][dim])

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Step computed in {elapsed_time:.9f} seconds.")

    plt.xlim(0,5)
    plt.ylim(-5,5)

    # Plotting vel and acc vectors
    plt.quiver(final["pos"][0], final["pos"][1], initial["acc"][0], initial["acc"][1], color='red', scale=15)  # Acceleration
    plt.quiver(final["pos"][0], final["pos"][1], final["vel"][0], final["vel"][1], color='blue', scale=15)  # Velocity

    # Plotting pos of particle
    plt.scatter(final["pos"][0], final["pos"][1], color="black")
    plt.pause(time_step/1.6)

loop_endtime = time.time()

print(f"Simulation lasted {loop_endtime - loop_time} seconds.")

plt.show()