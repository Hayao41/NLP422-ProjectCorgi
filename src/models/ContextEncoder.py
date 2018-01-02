import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    
    ''' stentence context encoder for sequential context '''
    
    def __init__(self, options):
        super(ContextEncoder, self).__init__()
        self.options = options

        # hidden states of lstms
        self.bi_hidden = (
            Variable(torch.zeros(2, 1, self.options.bi_hid_dims // 2)),
            Variable(torch.zeros(2, 1, self.options.bi_hid_dims // 2))
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

        # Bi_directional LSTM unit for represent contextual word vector
        #
        # @Dimension : context_linear_dim -> bi_hid_dims
        self.bi_lstm = nn.LSTM(
            self.options.context_linear_dim,
            self.options.bi_hid_dims // 2,
            bidirectional=True
        )

        # test tagging linear layer
        # make lstm out to be final predictions
        # context_linear_dim -> pos_vocab_size
        self.hidden2tag = nn.Linear(self.options.bi_hid_dims, self.options.pos_vocab_size)

        if options.xavier:
            self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal(self.context_linear.weight)

    def init_bi_hidden(self):

        '''
        initialize bi-lstm's hidden state and memory cell state\n
         '''

        self.bi_hidden = (
            Variable(torch.zeros(2, 1, self.options.bi_hid_dims // 2)),
            Variable(torch.zeros(2, 1, self.options.bi_hid_dims // 2))
        )

    def nonlinear_transform(self, graph):

        ''' nonlinear_transform for concatenated vector (word embeddings and pos embeddings) '''

        pos_embeddings = graph.pos_embeddings
        if self.options.word_emb_dims is not 0:
            word_embeddings = graph.word_embeddings
            cat_vectors = torch.cat((word_embeddings, pos_embeddings), 1)
            hidden_vectors = F.sigmoid(self.context_linear(cat_vectors))
        else:
            hidden_vectors = F.sigmoid(self.context_linear(pos_embeddings))

        return hidden_vectors

    def bi_lstm_transform(self, hidden_vectors, graph):
        
        lstm_out, self.bi_hidden = self.bi_lstm(
            hidden_vectors.view(len(graph.indexedWords), 1, -1),
            self.bi_hidden
        )
        
        graph.setContextVector(lstm_out.view(len(graph.indexedWords), -1))

    def test_bi_transform(self, hidden_vectors, graph):

        ''' pos tagging by bi_lstm '''

        self.init_bi_hidden()

        # feed word embeddings into bi_lstm
        lstm_out, self.bi_hidden = self.bi_lstm(
            hidden_vectors.view(len(graph.indexedWords), 1, -1),
            self.bi_hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(len(graph.indexedWords), -1))
        tag_score = F.log_softmax(tag_space)
        graph.setContextVector(lstm_out.view(len(graph.indexedWords), -1))
        return tag_score

    def forward(self, graph):

        self.init_bi_hidden()

        hidden_vectors = self.nonlinear_transform(graph)
        # hidden_vectors = graph.pos_embeddings
        # hidden_vectors = graph.word_embeddings
        self.bi_lstm_transform(hidden_vectors, graph)
        # return self.test_bi_transform(hidden_vectors, graph)
