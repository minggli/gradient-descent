from torch import nn
import torch

class TwoLayerNN(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.weight_1 = nn.Parameter(torch.empty(in_features, 4))
        self.bias_1 = nn.Parameter(torch.empty(4))
        self.weight_2 = nn.Parameter(torch.empty(4, 1))
        self.bias_2 = nn.Parameter(torch.empty(1))

        for param in self.parameters():
            torch.nn.init.normal_(param, mean=0, std=1)

    def forward(self, x):
        return self.func(x, self.weight_1, self.bias_1, self.weight_2, self.bias_2)

    def func(self, x, weight_1, bias_1, weight_2, bias_2):
        h_1 = x @ weight_1 + bias_1
        return h_1 @ weight_2 + bias_2
