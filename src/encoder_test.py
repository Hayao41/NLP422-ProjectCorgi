
from utils.Utils import options
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.Encoder import ContextEncoder
import utils.Utils as Utils
import torch.optim as optim
from matplotlib import pyplot as plt

options = options(
    word_vocab_size=30,
    word_emb_dims=20,
    pos_vocab_size=30,
    pos_emb_dims=0,
    rel_vocab_size=30,
    rel_emb_dims=0,
    context_linear_dim=20,
    use_bi_lstm=True,
    lstm_hid_dims=20,
    lstm_direction=2,
    lstm_num_layers=1,
    chain_hid_dims=20,
    batch_size=2,
    xavier=True,
    dropout=0.1,
    cuda=True
)

word1 = "I like this dog.".split()
word2 = "He loves that pen.".split()

POS = ["VV", "NN", "CON"]

word2idx = {element : i for i, element in enumerate(word1 + word2)}
pos2idx = {element : i for i, element in enumerate(POS)}

idx1 = [word2idx[w] for w in word1]
idx2 = [word2idx[w] for w in word2]

pos1 = [1, 0, 2, 1]
pos2 = [1, 0, 2, 1]

words = Variable(torch.LongTensor([idx1, idx2]))

pos = Variable(torch.LongTensor([pos1, pos2]))


class Sequences(object):
    
    def __init__(self, words=None, pos=None, batch_size=1):
        
        self.words = words
        self.pos = pos
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def switch2gpu(self):
        if self.words is not None:
            self.words = self.words.cuda()
        if self.pos is not None:
            self.pos = self.pos.cuda()

    def switch2cpu(self):
        if self.words is not  None:
            self.words = self.words.cpu()
        if self.pos is not None:
            self.pos = self.pos.cpu()
        

class TestModel(nn.Module):
    
    def __init__(self, options, encoder):
        super(TestModel, self).__init__()

        self.options = options

        self.encoder = encoder

        self.hid2tag = nn.Linear(self.options.lstm_hid_dims, len(pos2idx))

    def switch2gpu(self):
        self.encoder.switch2gpu()
        self.cuda()

    def forward(self, sequences):
        
        out = self.encoder(sequences)
        out = self.hid2tag(out)

        return F.log_softmax(out, dim=-1)


sequences = Sequences(words=words, pos=pos, batch_size=2)
encoder = ContextEncoder(options=options)
test = TestModel(options=options, encoder=encoder)
crit = nn.NLLLoss()
optimizer = optim.SGD(test.parameters(), lr=0.1, momentum=0.2)

if options.cuda:
    test.switch2gpu()
    sequences.switch2gpu()
    crit = crit.cuda()

print(sequences.words, sequences.pos)

e_list = []
l_list = []

for epoch in range(1000):
    e_list.append(epoch)

    test.zero_grad()
    out = test(sequences)

    loss = 0

    for inst in range(sequences.batch_size):
        # print(out[inst])
        # print(sequences.pos[inst])
        loss = loss + (1/sequences.batch_size) * crit(out[inst], sequences.pos[inst])

    l_list.append(loss.data[0])

    loss.backward()
    optimizer.step()

plt.plot(e_list, l_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

