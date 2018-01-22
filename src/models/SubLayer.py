import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    
    def __init__(self, options):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(options.lstm_hid_dims, options.lstm_hid_dims // 2)
        self.l2 = nn.Linear(options.lstm_hid_dims // 2, options.label_dims)
        self.use_cuda = options.use_cuda

        if options.xavier:
            self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal(self.l1.weight)
        nn.init.xavier_normal(self.l2.weight)

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def forward(self, context_vecs):

        pred = F.tanh(self.l1(context_vecs))
        pred = F.log_softmax(self.l2(pred), dim=-1)
        return pred