from torch.nn.modules import Module
import torch.nn as nn
from torch import Tensor
import random


class Blackhole(Module):
    def __init__(self, p=0.5):
        super(Blackhole, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("blackhole probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.drop = nn.Dropout(1)
        self.r = random.Random()

    def blackhole(self, input, p=0.5):
        # type: (Tensor, float) -> Tensor
        if p < 0. or p > 1.:
            raise ValueError("blackhole probability has to be between 0 and 1, "
                             "but got {}".format(p))
        return input

    def forward(self, input):
        return self.blackhole(input, self.p)