"""
bridge unit for parallel-network
"""
import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize with ones matrix
        self.bridge = nn.Parameter(torch.ones(2, 2))

    def forward(self, input1, input2):
        output1 = input1 * self.bridge[0][0] + input2 * self.bridge[0][1]
        output2 = input1 * self.bridge[1][0] + input2 * self.bridge[1][1]

        return output1, output2
