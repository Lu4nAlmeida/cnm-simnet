import numpy as np

class KinematicsSimulation:
    def __init__(self, pos, vel, acc):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.acc = np.array(acc)

    def compute_output(self, time_steps):
        self.pos = self.vel * time_steps + (self.acc * time_steps**2) / 2
        self.vel = self.vel + self.acc * time_steps

    def random_initial_parameters(self, num_dim, shape):
        self.pos = np.broadcast_to(np.zeros(num_dim), shape),
        self.vel = np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
        self.acc = np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),