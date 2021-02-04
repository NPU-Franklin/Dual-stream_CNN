"""
cross_stitch unit for parallel-network
"""
import torch
import torch.nn as nn


class CrossStitch(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        # initialize
        self.cross_stitch = nn.Parameter(torch.tensor([[alpha, beta],
                                                       [beta, alpha]]))

    def forward(self, input1, input2):
        output1 = input1 * self.cross_stitch[0][0] + input2 * self.cross_stitch[0][1]
        output2 = input2 * self.cross_stitch[1][0] + input1 * self.cross_stitch[1][1]

        return output1, output2
