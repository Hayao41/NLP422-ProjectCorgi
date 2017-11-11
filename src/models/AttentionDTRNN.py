import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionDTRNN(nn.Module):

    def __init__(self):
        super(AttentionDTRNN, self).__init__()

    def forward(self):
        print("forward pass")


a = AttentionDTRNN()
a.forward()




