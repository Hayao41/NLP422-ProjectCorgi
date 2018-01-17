
from utils.Utils import options
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.ContextEncoder import ContextEncoder
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
    l_hid_dims=5,
    r_hid_dims=5,
    batch_size=2,
    xavier=True,
    dropout=0.1,
    cuda=False
)

word1 = "I like this dog.".split()
word2 = "He loves that pen.".split()

POS = ["VV", "NN", "CON"]

word2idx = Utils.make_dictionary(word1+word2)
pos2idx = Utils.make_dictionary(POS)

idx1 = [word2idx[w] for w in word1]
idx2 = [word2idx[w] for w in word2]

pos1 = [1, 0, 2, 1]
pos2 = [1, 0, 2, 1]

words = Variable(torch.LongTensor([idx1, idx2]))

pos = Variable(torch.LongTensor([pos1, pos2]))

print(words, pos)


class Sequences(object):
    
    def __init__(self, words=None, pos=None, batch_size=1):
        
        self.words = words
        self.pos = pos
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def switch2gpu(self):
        self.words.cuda()
        self.pos.cuda()

    def switch2cpu(self):
        self.words.cpu()
        self.pos.cpu()
        

class TestModel(nn.Module):
    
    def __init__(self, options, encoder):
        super(TestModel, self).__init__()

        self.options = options

        self.encoder = encoder

        self.hid2tag = nn.Linear(self.options.lstm_hid_dims, 3)

    def forward(self, sequences):
        
        out = self.encoder(sequences)
        out = self.hid2tag(out)

        return F.log_softmax(out, dim=-1)


sequences = Sequences(words=words, pos=pos, batch_size=2)
encoder = ContextEncoder(options=options)
test = TestModel(options=options, encoder=encoder)

if options.cuda:
    encoder.switch2gpu()
    sequences.switch2gpu()

crit = nn.NLLLoss()
optimizer = optim.SGD(test.parameters(), lr=0.1, momentum=0.2)


e_list = []
l_list = []

for epoch in range(1000):
    e_list.append(epoch)

    test.zero_grad()
    out = test(sequences)

    loss = 0

    for inst in range(sequences.batch_size):
        loss = loss + (1/sequences.batch_size) * crit(out[inst], sequences.pos[inst])

    l_list.append(loss.data[0])

    loss.backward()
    optimizer.step()

sequences = Sequences(words=Variable(torch.LongTensor(idx1)))

print(test(sequences))

plt.plot(e_list, l_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

