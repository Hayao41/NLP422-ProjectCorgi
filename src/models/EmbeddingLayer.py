import torch
from torch.autograd import Variable
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    
    def __init__(self, options):

        super(EmbeddingLayer, self).__init__()

        self.options = options
        
        # POS embedding layer
        #
        # @Dimension : pos_vocab_size -> pos_emb_dims
        self.pos_emb_layer = nn.Embedding(
            self.options.pos_vocab_size,
            self.options.pos_emb_dims
        )

        # Arc_relation tag embedding layer
        #
        # @Dimension : rel_vocab_size -> rel_emb_dims
        self.rel_emb_layer = nn.Embedding(
            self.options.rel_vocab_size,
            self.options.rel_emb_dims
        )

        # Word embedding layer(randomly initialize or by word2vec)
        #
        # @Dimension : word_vocab_size -> word_emb_dims
        self.word_emb_layer = nn.Embedding(
            self.options.word_vocab_size,
            self.options.word_emb_dims
        )

        # self.position_emb_layer = nn.Embedding()

        if self.options.xavier:
            self.xavier_normal()

    def loadWeights(self):
        pass

    def xavier_normal(self):
        
        ''' xavier weights initialize '''

        if self.options.pos_emb_dims is not 0:
            nn.init.xavier_normal(self.pos_emb_layer.weight)

        if self.options.rel_emb_dims is not 0:
            nn.init.xavier_normal(self.rel_emb_layer.weight)

        if self.options.word_emb_dims is not 0:
            nn.init.xavier_normal(self.word_emb_layer.weight)

    def wordEmbedding(self, words):
        return self.word_emb_layer(words)

    def POSEmbedding(self, pos):
        return self.pos_emb_layer(pos)
        
    def relationEmbedding(self, graphs):
        
        for graph in graphs:
            rel_idxs = graph.getArcRelationIdxs()
            rel_embeddings = self.rel_emb_layer(rel_idxs)
            graph.setArcRelationEmbeddings(rel_embeddings)

    def forward(self, batch_data):
        
        sequences, graphs = batch_data
        words, pos = sequences

        wordEmbeddings =  self.wordEmbedding(words)
        POSEmbeddings = self.POSEmbedding(pos)
        self.relationEmbedding(graphs)

        return (wordEmbeddings, POSEmbeddings)
