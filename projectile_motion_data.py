import matplotlib.pyplot as plt
import numpy as np
import math
import time

# https://youtu.be/Tr_TpLk3dY8?si=ctMPCvV_lPWFMLKZ Ben Yelverton
# This simulation implements linear drag.
# Linear drag is only accurate for objects with Reynolds Numbers < 10

num_dim = 2
time_step = 0.01
total_time = 1.3


def drag_force(mass, acc, drg, vel, t=0):
    return -(drg * final_velocity(mass, acc, drg, vel, t))


def final_velocity(mass, acc, drg, vel, t):
    return (vel - (mass * acc)/drg) * math.e**(-(drg * t)/mass) + (mass * acc)/drg


def delta_position(mass, acc, drg, vel, t):
    return (mass/drg) * (vel - (mass * acc)/drg) * (1 - math.e**((-drg * t) / mass)) + (mass * acc * t) / drg


# v(t) = (v - mg/k) * e^(-kt/m) + mg/k
# x(t) = (m/k) * (v - (mg)/k) * (1 - e**(-(kt)/m)) + (mgt/k)


# Initial conditions
initial = {"pos": np.zeros(num_dim),
           "vel": np.array([5, 5]),
           "mass": 1.5,
           "acc": np.array([0, -9]),
           "drg": 0.6}

# Final conditions
final = {"pos": np.copy(initial["pos"]),
         "vel": np.copy(initial["vel"]),
         "frc": np.array([drag_force(initial["mass"], initial["acc"][0], initial["drg"], initial["vel"][0]),
                          drag_force(initial["mass"], initial["acc"][1], initial["drg"], initial["vel"][1])]
                        )
        }

loop_time = time.time()
# Main loop
for i in range(int(total_time / time_step)):
    plt.clf()
    delta_time = i * time_step  # Elapsed time

    # Measure start time
    start_time = time.perf_counter()
    for dim in range(num_dim):
        final["pos"][dim] = delta_position(initial["mass"], initial["acc"][dim], initial["drg"], initial["vel"][dim], delta_time) + initial["pos"][dim]
        final["vel"][dim] = final_velocity(initial["mass"], initial["acc"][dim], initial["drg"], initial["vel"][dim], delta_time)
        final["frc"][dim] = drag_force(initial["mass"], initial["acc"][dim], initial["drg"], initial["vel"][dim], delta_time)

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Step computed in {elapsed_time:.7f} seconds.")

    plt.xlim(-1, 9)
    plt.ylim(-5, 5)

    # Plotting vel, acc and drg vectors
    plt.quiver(final["pos"][0], final["pos"][1], final["frc"][0], final["frc"][1], color='orange', scale=15)  # Drag Force
    plt.quiver(final["pos"][0], final["pos"][1], final["vel"][0], final["vel"][1], color='blue', scale=15)  # Velocity
    plt.quiver(final["pos"][0], final["pos"][1], initial["acc"][0], initial["acc"][1], color='red', scale=15)  # Acceleration

    # Plotting pos of particle
    plt.scatter(final["pos"][0], final["pos"][1], color="black")
    plt.pause(time_step/1.6)

loop_endtime = time.time()

print(f"Simulation lasted {loop_endtime - loop_time} seconds.")

plt.show()

# PS.: It took me a total of 3 hours and a half hours of debugging to figure this out
