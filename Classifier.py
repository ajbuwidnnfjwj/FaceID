import torch
import torch.nn as nn
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out