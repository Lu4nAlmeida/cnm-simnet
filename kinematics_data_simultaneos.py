import matplotlib.pyplot as plt
import numpy as np
import time

# Parameters
num_dim = 2
time_step = 0.05
t = 5

# Initial conditions
initial = {"pos": np.random.uniform(-3, 3, num_dim),
           "vel": np.random.uniform(-3, 3, num_dim)}
acc = np.random.uniform(-3, 3, num_dim)

# Time steps as a NumPy array
delta_times = np.arange(0, t, time_step)

# Measure start time
start_time = time.perf_counter()

# Compute displacement and velocity for all delta_times
displacements = initial["vel"] * delta_times[:, None] + 0.5 * acc * (delta_times[:, None] ** 2)
final_velocities = initial["vel"] + acc * delta_times[:, None]

# Measure end time
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Simulation computed in {elapsed_time:.9f} seconds.")

loop_time = time.time()
# Visualization
for i, delta_time in enumerate(delta_times):
    plt.clf()

    pos = displacements[i]
    vel = final_velocities[i]

    # Set axis limits dynamically
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Plot velocity and acceleration vectors
    plt.quiver(pos[0], pos[1], acc[0], acc[1], color='red', scale=15)  # Acceleration
    plt.quiver(pos[0], pos[1], vel[0], vel[1], color='blue', scale=15)  # Velocity

    # Plot particle position
    plt.scatter(pos[0], pos[1], color="black")
    plt.pause(time_step/1.6)

loop_endtime = time.time()

print(f"Simulation lasted {loop_endtime - loop_time} seconds.")

plt.show()