import torch.optim
from torch import nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 512)
        self.l3 = nn.Linear(512, 128)
        self.l4 = nn.Linear(128, output_size)
        nn.init.xavier_normal(self.l1.weight)
        nn.init.xavier_normal(self.l2.weight)
        nn.init.xavier_normal(self.l3.weight)

    def forward(self, x):
        x = x.type(torch.float32)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


def get_optimizer(model, learning_rate=0.001):
    return torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def get_criterion():
    return nn.MSELoss()
