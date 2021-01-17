"""
bridge unit for parallel-network
"""
import torch
import torch.nn as nn


class Bridge(nn.Module):
    def __init__(self, input1, input2):
        super().__init__()
        self.input1 = input1
        self.input2 = input2
        self.input1_reshaped = torch.flatten(self.input1, start_dim=1, end_dim=-1)
        self.input2_reshaped = torch.flatten(self.input2, start_dim=1, end_dim=-1)
        self.input = torch.cat((self.input1_reshaped, self.input2_reshaped), 1)
        # initialize with identity matrix
        self.bridge = nn.Parameter(torch.eye(self.input.shape[1], self.input.shape[1]))

    def forward(self):
        output = torch.mm(self.input, self.bridge)

        # need to call .value to convert Dimension objects to normal value
        input1_shape = list(-1 if s.value is None else s.value for s in self.input1.shape)
        input2_shape = list(-1 if s.value is None else s.value for s in self.input2.shape)
        output1 = torch.reshape(output[:, :self.input1_reshaped.shape[1]], shape=input1_shape)
        output2 = torch.reshape(output[:, self.input1_reshaped.shape[1]:], shape=input2_shape)
        return output1, output2
