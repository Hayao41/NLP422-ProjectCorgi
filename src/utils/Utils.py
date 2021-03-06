import utils.Constants as Constants
import numpy as np
import torch
from torch.autograd import Variable


class options(object):
        
    ''' 
    hyperparameters
    '''

    def __init__(self,

                 # vocab size
                 word_vocab_size=0,
                 pos_vocab_size=0,
                 rel_vocab_size=0,
                 rp_vocab_size=0,
                 label_dims=0,

                 # embedding layer params
                 word_emb_dims=0,
                 pos_emb_dims=0,
                 rel_emb_dims=0,
                 rp_emb_dims=0,

                 # non linear trans
                 use_non_linear_trans=True,
                 context_linear_dim=0,
                 inner_hidden_dims=0,

                 # context encoder
                 use_bi_lstm=True,
                 lstm_num_layers=1,
                 lstm_hid_dims=0,


                 # tree children chain
                 use_bi_chain=False,
                 chain_num_layers=1,
                 chain_hid_dims=0,

                 # tree encoder type
                 #@Type DRN : Dynamic recursvie neural nets
                 #@Type HTLstms : Hierarchical Tree LSTMs
                 use_tree=True,
                 tree_type="DRN",
                 direction="B2T",

                 # attention
                 atten_type="general",

                 # optimization
                 xavier=True,
                 train_batch_size=1,
                 eval_batch_size=1,
                 epoch=30,
                 dropout=0.,
                 padding=0,
                 use_cuda=False,
                 cuda_device="1",
                 optim="Adam",
                 lr=0.0001,
                 lr_decay=0.1,
                 weight_decay=0.0001,
                 momentum=0.5,
                 betas=(0.9, 0.98),
                 eps=1e-9,
                 loss_reduce=True,
                 sample_prop=-1,
                 sample_mode="down",
                 save_model=True,
                 save_mode="all",
                 model_path="../src/model_state/",
                 log_path="../src/log/",
                 pic_path="../src/visual/",

                 # data set prop
                 train_prop=0.7,
                 test_prop=0.3,
                 dev_prop=0.0
                ):
        super(options, self).__init__()

        assert sample_prop == -1 or sample_prop > 0, "sample_prop should be bigger than 0 or equals to -1(no scaling)"

        # ============ vocabubary size ============ #
        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.rel_vocab_size = rel_vocab_size
        self.rp_vocab_size = rp_vocab_size
        self.label_dims = label_dims

        # ============ embeddding layer ============ #
        self.word_emb_dims = word_emb_dims
        self.pos_emb_dims = pos_emb_dims
        self.rel_emb_dims = rel_emb_dims
        self.rp_emb_dims = rp_emb_dims

        # ============ non linear trans ============ #
        self.use_non_linear_trans = use_non_linear_trans
        self.context_linear_dim = context_linear_dim
        self.inner_hidden_dims = inner_hidden_dims

        # ========== context encoder(lstm) ========== #
        self.use_bi_lstm = use_bi_lstm
        if use_bi_lstm:
            self.lstm_direction = 2
        else:
            self.lstm_direction = 1
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hid_dims = lstm_hid_dims

        # ======== tree children chain(lstm) ======= #
        self.use_bi_chain = use_bi_chain
        if use_bi_chain:
            self.chain_direction = 2
        else:
            self.chain_direction = 1
        self.chain_num_layers = chain_num_layers
        self.chain_hid_dims = chain_hid_dims

        # ============ tree encoder type =========== #
        self.use_tree = use_tree
        self.tree_type = tree_type
        self.direction = direction

        # ============== attention ============== #
        self.atten_type = atten_type

        # ============ optimization ============ #
        self.xavier = xavier
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epoch = epoch
        self.dropout = dropout
        self.padding = padding
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.optim = optim
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.loss_reduce = loss_reduce
        self.sample_prop = sample_prop
        self.sample_mode = sample_mode
        self.save_model = save_model
        self.save_mode = save_mode
        self.model_path = model_path
        self.log_path = log_path
        self.pic_path = pic_path

        # =========== data set prop =========== #
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.dev_prop = dev_prop


def make_dictionary(vocab_list, pad={}):
    
    '''
    build index dictionary for vocabulary
    '''

    idxs = pad

    for element in vocab_list:
        if element not in idxs:
            idxs[element] = len(idxs)

    return idxs


def repackage_hidden(h):
    
    """Wraps hidden states in new Variables, to detach them from their history."""
    
    if type(h) == Variable:
        new_h = Variable(h.data)
        new_h.zero_()
        del h
        return new_h

    else:
        new_tuple = tuple(repackage_hidden(v) for v in h)
        del h
        return new_tuple
