import numpy as np
import math
import random

class ProjectileMotionSimulation:
    def __init__(self, pos, vel, mass, acc, drg):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.mass = mass
        self.acc = np.array(acc)
        self.drg = drg
        self.frc = self.mass * self.acc

    def compute_output(self, time_steps):
        self.pos = self.delta_position(self.mass, self.acc, self.drg, self.vel, time_steps)
        self.vel = self.final_velocity(self.mass, self.acc, self.drg, self.vel, time_steps)
        self.frc = self.drag_force(self.mass, self.acc, self.drg, self.vel, time_steps)

    def random_initial_parameters(self, num_dim, shape):
        self.pos = np.broadcast_to(np.zeros(num_dim), shape),
        self.vel = np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
        self.mass = np.full(shape, random.uniform(0, 10)),
        self.acc = np.broadcast_to(np.random.uniform(-10, 10, num_dim), shape),
        self.drg = np.full(shape, random.uniform(0, 5))

    @staticmethod
    def final_velocity(mass, acc, drg, vel, t):
        return (vel - (mass * acc) / drg) * math.e ** (-(drg * t) / mass) + (mass * acc) / drg

    @staticmethod
    def delta_position(mass, acc, drg, vel, t):
        return (mass / drg) * (vel - (mass * acc) / drg) * (1 - math.e ** ((-drg * t) / mass)) + (mass * acc * t) / drg

    @staticmethod
    def drag_force(mass, acc, drg, vel, t=0):
        return -(drg * ProjectileMotionSimulation.final_velocity(mass, acc, drg, vel, t))