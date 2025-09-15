import torch.nn as nn


class MultiClassNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 layer_sizes: list[int],
                 num_classes: int):
        super(MultiClassNN, self).__init__()
        layers = []
        current_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(nn.ReLU())
            current_dim = size
        layers.append(nn.Linear(current_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)