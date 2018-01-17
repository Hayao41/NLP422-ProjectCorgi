
from utils.Utils import options
import torch
from torch.autograd import Variable
from models.ContextEncoder import ContextEncoder
import numpy as np

options = options(
    word_vocab_size=30,
    word_emb_dims=20,
    pos_vocab_size=30,
    pos_emb_dims=0,
    rel_vocab_size=30,
    rel_emb_dims=0,
    context_linear_dim=20,
    lstm_hid_dims=20,
    lstm_direction=2,
    lstm_num_layers=1,
    l_hid_dims=5,
    r_hid_dims=5,
    xavier=True,
    batch_size=1,
    use_bi_lstm=True
)


class Sequence(object):
    
    def __init__(self, words=None, pos=None, WordEmbeddings=None, PosEmbeddings=None):
        
        self.words = words
        self.pos = pos
        self.WordEmbeddings = WordEmbeddings
        self.PosEmbeddings = PosEmbeddings


word = Variable(torch.from_numpy(np.random.randint(5, size=(options.batch_size, 10))))
pos = Variable(torch.from_numpy(np.random.randint(5, size=(options.batch_size, 10))))

WordEmbeddings = Variable(torch.zeros(1))
PosEmbeddings = Variable(torch.zeros(1))

sequence = Sequence(words=word,pos=pos, WordEmbeddings=WordEmbeddings, PosEmbeddings=PosEmbeddings)

encoder = ContextEncoder(options=options)

out = encoder(sequence)

print(out.view(-1, 10, options.lstm_hid_dims))
