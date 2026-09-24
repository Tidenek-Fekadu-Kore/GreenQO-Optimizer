import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),# Layer 1: W1 * x_norm + b1
            nn.ReLU(),# Activation 1 -> produces h1
            nn.Linear(64, 32),# Layer 2: W2 * h1 + b2
            nn.ReLU(),# Activation 2 -> produces h2
            nn.Linear(32, 1)# Layer 3: W3 * h2 + b3 -> produces y_hat
        )

    def forward(self, x):
        return self.net(x)