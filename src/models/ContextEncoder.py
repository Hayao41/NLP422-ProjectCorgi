import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    
    ''' sentence context rnn encoder '''
    
    def __init__(self, options):
        
        super(ContextEncoder, self).__init__()
        self.options = options

        # Sequence words embedding
        self.word_embeddings = nn.Embedding(
            self.options.word_vocab_size,
            self.options.word_emb_dims
        )

        # Sequence pos embedding
        self.pos_embeddings = nn.Embedding(
            self.options.pos_vocab_size,
            self.options.pos_emb_dims
        )

        # Encoding every word in context concatenated vector linear layer
        #
        # vi = g(W (word_emb con pos_emb) + b)
        #   where con is the concatenate operator
        #
        # @Dimension : word_emb_dims + pos_emb_dims -> context_linear_dim
        self.context_linear = nn.Linear(
            self.options.word_emb_dims + self.options.pos_emb_dims,
            self.options.context_linear_dim,
            bias=True
        )

        # LSTM unit for represent contextual word vector
        #
        # @Dimension : context_linear_dim -> lstm_hid_dims
        self.lstm = nn.LSTM(
            self.options.context_linear_dim,
            self.options.lstm_hid_dims // self.options.lstm_direction,
            bidirectional=self.options.use_bi_lstm
        )

        if options.xavier:
            self.xavier_normal()
    
    def xavier_normal(self):
        ''' xavier weights normalization '''

        nn.init.xavier_normal(self.context_linear.weight)

        if self.options.word_emb_dims is not 0:
            nn.init.xavier_normal(self.word_embeddings.weight)
        
        if self.options.pos_emb_dims is not 0:
            nn.init.xavier_normal(self.pos_embeddings.weight)

    def embedding(self, sequences):
        ''' sequence word or pos embedding '''
        
        if self.options.word_emb_dims is not 0:
            
            words_shape_size = len(sequences.words.data.shape)
            assert 0 < words_shape_size < 3, 'out of shape size, expected less than 3 and bigger than 0 but got {}'.format(words_shape_size)

            sequences.WordEmbeddings = self.word_embeddings(sequences.words)

        if self.options.pos_emb_dims is not 0:
            
            pos_shape_size = len(sequences.pos.data.shape)
            assert 0 < pos_shape_size < 3, 'out of shape size, expected less than 3 and bigger than 0 but got {}'.format(pos_shape_size)

            sequences.PosEmbeddings = self.pos_embeddings(sequences.pos)

    def nonlinear_transformation(self, sequences):
    
        ''' 
        nonlinear transformation for embedding vector (word embeddings, pos embeddings or word and pos cat vector)\n 
        @CatTrans : vi = g(W (word_emb con pos_emb) + b)\n
        @WordTrans : vi = g(W word_emb + b)\n
        @PosTrans : vi = g(W pos_emb + b)\n
        '''

        if min(self.options.word_emb_dims, self.options.pos_emb_dims) is not 0:
            cat_vectors = torch.cat((sequences.WordEmbeddings, sequences.PosEmbeddings), 2)
            input_vectors = F.tanh(self.context_linear(cat_vectors))

        elif self.options.word_emb_dims is not 0:
            input_vectors = F.tanh(self.context_linear(sequences.WordEmbeddings))
            
        else:
            input_vectors = F.tanh(self.context_linear(sequences.PosEmbeddings))

        return input_vectors

    def lstm_transformation(self, input_vectors):
        
        ''' 
        lstm transformation to get context vector \n
        @Trans : \n
        htl = LSTM(xt, ht-1) (-->)\n
        ht = htl\n
        @If use_bi_lstm is true\n
        htr = LSTM(xt-1, ht) (<--)\n
        ht = htl con htr\n
        '''

        def getInitHidden(cuda=False):
            ''' initialize hidden and memory cell state of lstm at the first time step'''

            if self.options.cuda:
                return (
                    Variable(torch.zeros(
                        self.options.lstm_num_layers * self.options.lstm_direction, 
                        self.options.batch_size, 
                        self.options.lstm_hid_dims // self.options.lstm_direction)
                    ).cuda(),
                    Variable(torch.zeros(
                        self.options.lstm_num_layers * self.options.lstm_direction, 
                        self.options.batch_size, 
                        self.options.lstm_hid_dims // self.options.lstm_direction)
                    ).cuda()
                )

            else:
                return (
                    Variable(torch.zeros(
                        self.options.lstm_num_layers * self.options.lstm_direction, 
                        self.options.batch_size, 
                        self.options.lstm_hid_dims // self.options.lstm_direction)
                    ),
                    Variable(torch.zeros(
                        self.options.lstm_num_layers * self.options.lstm_direction, 
                        self.options.batch_size, 
                        self.options.lstm_hid_dims // self.options.lstm_direction)
                    )
                )
        
        hidden_state = getInitHidden(cuda=self.options.cuda)
        
        # lstm contextual encoding
        lstm_out, hidden_state = self.lstm(
            input_vectors.view(
                -1, # batch sentence length
                self.options.batch_size, # batch_size
                self.options.context_linear_dim # input vector size
            ),
            hidden_state
        )

        return lstm_out

    def forward(self, sequences):
        
        self.embedding(sequences)

        input_vectors = self.nonlinear_transformation(sequences)

        return self.lstm_transformation(input_vectors)
