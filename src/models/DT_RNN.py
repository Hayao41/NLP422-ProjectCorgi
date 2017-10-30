import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DT_RNN(nn.Module):

    def __init__(self):
        super(DT_RNN, self).__init__()

    def forward(self):
        print("forward pass")


a = DT_RNN()
a.forward()




