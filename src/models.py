import torch
import torch.nn as nn

class ProjectileMotionModel(nn.Module):
    def __init__(self, in_dim=5, out_dim=4, hidden_units=256, activation=nn.ReLU()):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim, hidden_units), # Input Layer
            activation,
            nn.Linear(hidden_units, hidden_units), # Hidden Layer 1
            activation,
            nn.Linear(hidden_units, hidden_units), # Hidden Layer 2
            activation,
            nn.Linear(hidden_units, out_dim) # Output Layer
        )

    def forward(self, x):
        return self.net(x)


class ProjectileMotionModelRNN(nn.Module):
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
