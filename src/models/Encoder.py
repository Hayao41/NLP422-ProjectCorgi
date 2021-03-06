import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.Utils import repackage_hidden


class EmbeddingEncoder(nn.Module):

    ''' 
    Embedding layer for encode sparse input data like word, pos. This module 
    just encode word and part of speech tag(all are optional, controlled by options)\n
    @Attributes:\n
    word_embeddings: sentence word embedding(optional)\n
    pos_embeddings: sentence pos tag embedding(optional)\n
    context_linear: embedding vector transformation function\n
    '''
    
    def __init__(self, options):
        super(EmbeddingEncoder, self).__init__()
        # ============Hyperparameters==================#

        self.context_linear_dim = options.context_linear_dim
        self.word_emb_dims = options.word_emb_dims
        self.pos_emb_dims = options.pos_emb_dims
        self.use_non_linear_trans = options.use_non_linear_trans
        self.use_cuda = options.use_cuda
        self.padding_idx = options.padding

        assert (self.word_emb_dims is not 0) or (self.pos_emb_dims is not 0), "[Error] word dims and pos dims " \
                                                                              "are all 0, no input for nn!"
        
        if options.word_emb_dims is not 0:
            # Sequences words embedding
            self.word_embeddings = nn.Embedding(
                options.word_vocab_size,
                options.word_emb_dims,
                padding_idx=options.padding
            )

        if options.pos_emb_dims is not 0:
            # Sequences pos embedding
            self.pos_embeddings = nn.Embedding(
                options.pos_vocab_size,
                options.pos_emb_dims,
                padding_idx=options.padding
            )

        # Encoding every word in context concatenated vector linear layer
        #
        # vi = g(W (word_emb con pos_emb) + b)
        #   where con is the concatenate operator
        #
        # @Dimension : word_emb_dims + pos_emb_dims -> context_linear_dim
        if self.use_non_linear_trans:
            self.context_linear = nn.Linear(
                options.word_emb_dims + options.pos_emb_dims,
                options.context_linear_dim
            )

        if options.xavier:
            self.xavier_normal()

    def xavier_normal(self):
        ''' xavier weights normalization '''

        if self.use_non_linear_trans:
            nn.init.xavier_normal(self.context_linear.weight)

        if self.word_emb_dims is not 0:
            nn.init.xavier_normal(self.word_embeddings.weight)
            self.word_embeddings.weight.data[self.padding_idx].fill_(0)

        if self.pos_emb_dims is not 0:
            nn.init.xavier_normal(self.pos_embeddings.weight)
            self.pos_embeddings.weight.data[self.padding_idx].fill_(0)
    
    def embedding(self, sequences):
        ''' sequence word or pos embedding '''

        WordEmbeddings = None
        PosEmbeddings = None

        if self.word_emb_dims is not 0:
            
            words_shape_size = len(sequences.words_tensor.data.shape)
            assert 0 < words_shape_size < 3, 'out of shape range, expected less than 3 and bigger' \
                                             ' than 0 but got {}'.format(words_shape_size)

            WordEmbeddings = self.word_embeddings(sequences.words_tensor)

        if self.pos_emb_dims is not 0:
            
            pos_shape_size = len(sequences.pos_tensor.data.shape)
            assert 0 < pos_shape_size < 3, 'out of shape range, expected less than 3 and bigger ' \
                                           'than 0 but got {}'.format(pos_shape_size)

            PosEmbeddings = self.pos_embeddings(sequences.pos_tensor)

        return WordEmbeddings, PosEmbeddings

    def nonlinear_transformation(self, WordEmbeddings=None, PosEmbeddings=None):
        
        ''' 
        nonlinear transformation for embedding vector (word embeddings, pos embeddings or word and pos cat vector)\n 
        @CatTrans : vi = g(W (word_emb con pos_emb) + b)\n
        @WordTrans : vi = g(W word_emb + b)\n
        @PosTrans : vi = g(W pos_emb + b)\n
        '''
        
        if WordEmbeddings is not None and PosEmbeddings is not None:
            
            # cat word embeddings and pos embeddings along the last dimension
            cat_vectors = torch.cat((WordEmbeddings, PosEmbeddings), -1)

            input_vectors = F.tanh(self.context_linear(cat_vectors))

        elif WordEmbeddings is not None:
            input_vectors = F.tanh(self.context_linear(WordEmbeddings))
            
        else:
            input_vectors = F.tanh(self.context_linear(PosEmbeddings))

        return input_vectors

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def forward(self, sequences):
        
        WordEmbeddings, PosEmbeddings = self.embedding(sequences)
        
        if self.use_non_linear_trans:
            input_vectors = self.nonlinear_transformation(WordEmbeddings, PosEmbeddings)
            return input_vectors
        else:
            return WordEmbeddings, PosEmbeddings


class ContextEncoder(nn.Module):
    
    ''' 
    Sentence context rnn encoder(use lstm) before run it you 
    should call init_hidden() method to initialize lstm's hidden states. 
    During training or evaluating, you should call repackage_hidden() to repack 
    the last times hidden states to seprate it from last history\n
    @Attrubute\n
    embedding encoder: embedding layer for encode sparse input data\n
    lstm: Long Short Term Memory recurrent nets(bi-direction optional)\n
    '''
    
    def __init__(self, options):
        
        super(ContextEncoder, self).__init__()

        #============Hyperparameters==================#

        self.context_linear_dim = options.context_linear_dim
        self.lstm_hid_dims = options.lstm_hid_dims
        self.use_cuda = options.use_cuda
        self.batch_size = 1  # default batch size 1 (inference stage)

        # lstm initial state params
        self.total_layers = options.lstm_num_layers * options.lstm_direction
        self.single_pass_dims = self.lstm_hid_dims // options.lstm_direction

        # word embedding and pos embedding layer
        self.embedding_encoder = EmbeddingEncoder(options)

        # LSTM unit for represent contextual word vector
        #
        # @Dimension : context_linear_dim -> lstm_hid_dims
        self.lstm = nn.LSTM(
            options.context_linear_dim,
            options.lstm_hid_dims // options.lstm_direction,
            num_layers=options.lstm_num_layers,
            dropout=options.dropout,
            bidirectional=options.use_bi_lstm
        )

        self.hidden_state = None
    
    def lstm_transformation(self, input_vectors, hidden_state):
        
        ''' 
        lstm transformation to get context vector \n
        @Trans : \n
        htl = LSTM(xt, ht-1) (-->)\n
        ht = htl\n
        @If use_bi_lstm is true\n
        htr = LSTM(xt-1, ht) (<--)\n
        ht = htl con htr\n 
        '''
        
        # lstm contextual encoding
        lstm_out, new_hidden_state = self.lstm(
            input_vectors.view(
                -1,  # batch sentence length
                self.batch_size,  # train_batch_size
                self.context_linear_dim  # input vector size
            ),
            hidden_state
        )

        return lstm_out, new_hidden_state

    def init_hidden(self, batch_size):
        
        ''' init two_branch_chain_lstm's hidden state and memory cell state '''
        
        if self.use_cuda:
            self.hidden_state = (
                Variable(torch.zeros(
                    self.total_layers,
                    batch_size,
                    self.single_pass_dims)
                ).cuda(),
                Variable(torch.zeros(
                    self.total_layers,
                    batch_size,
                    self.single_pass_dims)
                ).cuda()
            )

        else:
            self.hidden_state = (
                Variable(torch.zeros(
                    self.total_layers,
                    batch_size,
                    self.single_pass_dims)
                ),
                Variable(torch.zeros(
                    self.total_layers,
                    batch_size,
                    self.single_pass_dims)
                )
            )

    def repackage_hidden(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        self.hidden_state = repackage_hidden(self.hidden_state)

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def forward(self, sequences):

        self.batch_size = len(sequences)

        input_vectors = self.embedding_encoder(sequences)
        
        out, self.hidden_state = self.lstm_transformation(input_vectors, self.hidden_state)

        return out.view(self.batch_size, -1, self.lstm_hid_dims)
