import torch
import torch.nn as nn
import numpy as np
import math

class ProjectileMotionModel(nn.Module):
    def __init__(self, input_features=7, output_features=6, hidden_units=64, activation=nn.ReLU()):
        '''
        Args:
          input_features: number of input neurons (num_dim * 2 + 3)
          output_features: number of output neurons (num_dim * 3)
          hidden_units: number of neurons per hidden layer

        Returns:
          A PyTorch model that predicts the motion of a particle based on set initial conditions.
        '''
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=hidden_units), # Input Layer
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units), # Hidden Layer 1
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units), # Hidden Layer 2
            activation,
            torch.nn.Linear(in_features=hidden_units, out_features=output_features) # Output Layer
        )

    def forward(self, x):
        return self.net(x)


# Create Model
class ProjectileMotionModelRNN(torch.nn.Module):
    def __init__(self, in_dim=6, hid_dim=128, num_layers=2, out_dim=6, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_dim,
                            hidden_size=hid_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, out_dim)
        )
    def forward(self, x):
        # x: (batch, seq, in_dim)
        seq_out, _ = self.rnn(x)          # (batch, seq, hid_dim)
        return self.head(seq_out)         # (batch, seq, out_dim)



class ProjectileMotionSimulation:
    def __init__(self, pos, vel, mass, acc, drg):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.mass = mass
        self.acc = np.array(acc)
        self.drg = drg
        self.frc = self.mass * self.acc

    def compute_output(self, initial, delta_times):
        self.pos = self.delta_position(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)
        self.vel = self.final_velocity(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)
        self.frc = self.drag_force(initial.mass, initial.acc, initial.drg, initial.vel, delta_times)

    @staticmethod
    def final_velocity(mass, acc, drg, vel, t):
        return (vel - (mass * acc) / drg) * math.e ** (-(drg * t) / mass) + (mass * acc) / drg

    @staticmethod
    def delta_position(mass, acc, drg, vel, t):
        return (mass / drg) * (vel - (mass * acc) / drg) * (1 - math.e ** ((-drg * t) / mass)) + (mass * acc * t) / drg

    @staticmethod
    def drag_force(mass, acc, drg, vel, t=0):
        return -(drg * ProjectileMotionSimulation.final_velocity(mass, acc, drg, vel, t))
