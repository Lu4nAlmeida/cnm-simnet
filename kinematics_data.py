import matplotlib.pyplot as plt
import numpy as np
import time

num_dim = 2
time_step = 0.05
duration = 5

# Initial conditions
initial = {"pos": np.zeros(num_dim),
           "vel": np.array([1, 3])}

acc = np.array([0, -2])

# Final conditions
final = {"pos": np.zeros(num_dim),
         "vel": np.zeros(num_dim)}


def displacement(vel, acc, time):
    return vel * time + (acc * time ** 2) * 0.5  # Kinematic equation for displacement


def final_velocity(vel, acc, time):
    return vel + acc * time  # Kinematic equation for final velocity


loop_time = time.time()
# Main loop
for i in range(int(duration / time_step)):
    plt.clf()
    delta_time = i * time_step  # Elapsed time

    # Measure start time
    start_time = time.perf_counter()
    for dim in range(num_dim):
        final["pos"][dim] = displacement(initial["vel"][dim], acc[dim], delta_time)
        final["vel"][dim] = final_velocity(initial["vel"][dim], acc[dim], delta_time)

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Step computed in {elapsed_time:.9f} seconds.")

    plt.xlim(0,5)
    plt.ylim(-5,5)

    # Plotting vel and acc vectors
    plt.quiver(final["pos"][0], final["pos"][1], acc[0], acc[1], color='red', scale=15)  # Acceleration
    plt.quiver(final["pos"][0], final["pos"][1], final["vel"][0], final["vel"][1], color='blue', scale=15)  # Velocity

    # Plotting pos of particle
    plt.scatter(final["pos"][0], final["pos"][1], color="black")
    plt.pause(time_step/1.6)

loop_endtime = time.time()

print(f"Simulation lasted {loop_endtime - loop_time} seconds.")

plt.show()