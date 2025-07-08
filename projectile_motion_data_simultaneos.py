import matplotlib.pyplot as plt
import numpy as np
import math
import time

# https://youtu.be/Tr_TpLk3dY8?si=ctMPCvV_lPWFMLKZ Ben Yelverton
# This simulation implements linear drag.
# Linear drag is only accurate for objects with Reynolds Numbers < 10

num_dim = 2
time_step = 0.01
duration = 1.3
shape = (int(duration / time_step), num_dim)  # Shape for each input

def drag_force(mass, acc, drg, vel, t=0):
    return -(drg * final_velocity(mass, acc, drg, vel, t))


def final_velocity(mass, acc, drg, vel, t):
    return (vel - (mass * acc)/drg) * math.e**(-(drg * t)/mass) + (mass * acc)/drg


def delta_position(mass, acc, drg, vel, t):
    return (mass/drg) * (vel - (mass * acc)/drg) * (1 - math.e**((-drg * t) / mass)) + (mass * acc * t) / drg


# v(t) = (v - mg/k) * e^(-kt/m) + mg/k
# x(t) = (m/k) * (v - (mg)/k) * (1 - e**(-(kt)/m)) + (mgt/k)

# Time steps as a NumPy array
delta_times = np.arange(0, duration, time_step)
delta_times = np.broadcast_to(delta_times[:,None], shape)          # Shape: (130, 2)

# Initial conditions
initial = {"pos": np.broadcast_to(np.zeros(num_dim), shape),
           "vel": np.broadcast_to(np.array([5, 5]), shape),
           "mass": np.full(shape, 1.5),
           "acc": np.broadcast_to(np.array([0, -9]), shape),
           "drg": np.full(shape, 0.6)}

# Final conditions
final = {"pos": np.copy(initial["pos"]),
         "vel": np.copy(initial["vel"]),
         "frc": np.array([drag_force(initial["mass"], initial["acc"][0], initial["drg"], initial["vel"][0]),
                          drag_force(initial["mass"], initial["acc"][1], initial["drg"], initial["vel"][1])]
                        )
        }

# Measure start time
start_time = time.perf_counter()

final["pos"] = delta_position(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times) + initial["pos"]
final["vel"] = final_velocity(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times)
final["frc"] = drag_force(initial["mass"], initial["acc"], initial["drg"], initial["vel"], delta_times)

# Measure end time
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Simulation computed in {elapsed_time:.9f} seconds.")

loop_time = time.time()
# Visualization
for i, delta_time in enumerate(delta_times):
    plt.clf()

    pos = final["pos"][i]
    vel = final["vel"][i]
    frc = final["frc"][i]

    # Set axis limits dynamically
    plt.xlim(-1, 9)
    plt.ylim(-5, 5)

    # Plot velocity and acceleration vectors
    plt.quiver(pos[0], pos[1], frc[0], frc[1], color='orange', scale=15)  # Drag Force
    plt.quiver(pos[0], pos[1], initial["acc"][0][0], initial["acc"][0][1], color='red', scale=15)  # Acceleration
    plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

    # Plot particle position
    plt.scatter(pos[0], pos[1], color="black")
    plt.pause(time_step/1.6)

loop_endtime = time.time()

print(f"Simulation lasted {loop_endtime - loop_time} seconds.")

plt.show()

# Around 2x slower than kinematics simultaneous
