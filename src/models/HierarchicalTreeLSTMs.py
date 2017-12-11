# -*- coding: utf-8 -*-
'''
This file define HierarchicalTreeLSTMs model
 '''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SemanticStructure as sstrut


class HierarchicalTreeLSTMs(nn.Module):
    '''
    Model implements HierarchicalTreeLSTMs(Yoav et al., 2016, https://arxiv.org/abs/1603.00375)\n
    This class contains three lstm unit for supporting hierarchical lstms for encoding a given
    dependency parse tree.\n
    @Author JoelChen\n
    @Time   2017/10/31\n
     '''

    def __init__(self, options=None):
        super(HierarchicalTreeLSTMs, self).__init__()

        if options is not None:

            # parameters
            self.pos_vocab_size = options.pos_vocab_size
            self.pos_emb_dims = options.pos_emb_dims
            self.rel_vocab_size = options.rel_vocab_size
            self.rel_emb_dims = options.rel_emb_dims
            self.word_vocab_size = options.word_vocab_size
            self.word_emb_dims = options.word_emb_dims
            self.vec_dims = options.vec_dims
            self.bi_hid_dims = options.bi_hid_dims
            self.l_hid_dims = options.l_hid_dims
            self.r_hid_dims = options.r_hid_dims
            self.rel_labeled_tag = options.rel_labeled_tag

            # hidden states of lstms
            self.bi_hidden = self.init_bi_hidden()
            self.left_hidden = self.init_left_hidden()
            self.right_hidden = self.init_right_hidden()

            # POS embedding layer
            #
            # @Dimension : pos_vocab_size -> pos_emb_dims
            self.pos_emb_layer = nn.Embedding(self.pos_vocab_size,
                                              self.pos_emb_dims)

            # Arc_relation tag embedding layer
            #
            # @Dimension : rel_vocab_size -> rel_emb_dims
            self.rel_emb_layer = nn.Embedding(self.rel_vocab_size,
                                              self.rel_emb_dims)

            # Word embedding layer(randomly initialize or by word2vec)
            #
            # @Dimension : word_vocab_size -> word_emb_dims
            self.word_emb_layer = nn.Embedding(self.word_vocab_size,
                                               self.word_emb_dims)

            # Encoding every word in context concatenated vector linear layer
            #
            # vi = g(W (word_emb con pos_emb) + b)
            #   where con is the concatenate operator
            #
            # @Dimension : word_emb_dims + pos_emb_dims -> vec_dims
            self.vec_linear = nn.Linear(self.word_emb_dims + self.pos_emb_dims,
                                        self.vec_dims,
                                        bias=True)

            # Bi_directional LSTM unit for represent contextual word vector
            #
            # @Dimension : vec_dims -> bi_hid_dims
            self.bi_lstm = nn.LSTM(self.vec_dims,
                                   self.bi_hid_dims // 2,
                                   bidirectional=True)

            # Encoding children concatenated vector linear layer
            #
            # enc(t) = g(W( el(t) con er(t) ) + b)
            #   where con is the concatenate operator
            #
            # if arc_relation representation is needed, the encoding function is changed to
            # enc(t) = g(W( el(t) con er(t) con rel_emb) + b)
            #   where con is the concatenate operator
            #
            # specially for leaf
            # enc(leaf) = g(W( el(leaf) con er(leaf) ) + b)
            #
            # @Dimension : l_hid_dims + r_hid_dims -> bi_hid_dims
            #   or l_hid_dims + r_hid_dims + rel_emb_dims -> bi_hid_dims
            if self.rel_labeled_tag is not None:
                self.enc_linear = nn.Linear(self.l_hid_dims + self.r_hid_dims + self.rel_emb_dims,
                                            self.bi_hid_dims,
                                            bias=True)
            else:
                self.enc_linear = nn.Linear(self.l_hid_dims + self.r_hid_dims,
                                            self.bi_hid_dims,
                                            bias=True)

            # LSTM unit computing left children vector
            #
            # el(t) = RNNl(vi(t), enc(t.l1), enc(t.l2), ... , enc(t.lk))
            #
            # specially for leaf
            # el(leaf) = RNNl(vi(leaf))
            #
            # @Dimension : bi_hid_dims -> l_hid_dims
            self.l_lstm = nn.LSTM(self.bi_hid_dims,
                                  self.l_hid_dims)

            # LSTM unit computing right children vector
            #
            # er(t) = RNNr(vi(t), enc(t.r1), enc(t.r2), ... , enc(t.rk))
            #
            # specially for leaf
            # er(leaf) = RNNr(vi(leaf))
            #
            # @Dimensions : bi_hid_dims -> r_hid_dims
            self.r_lstm = nn.LSTM(self.bi_hid_dims,
                                  self.r_hid_dims)

    def init_bi_hidden(self):
        '''
        initialize bi-lstm's hidden state and memory cell state\n
         '''
        return (
            Variable(torch.zeros(2, 1, self.bi_hid_dims // 2)),
            Variable(torch.zeros(2, 1, self.bi_hid_dims // 2))
        )

    def init_left_hidden(self):
        '''
        initialize left-chain-lstm's hidden state and memory cell state\n
         '''

        return (
            Variable(torch.zeros(1, 1, self.l_hid_dims)),
            Variable(torch.zeros(1, 1, self.l_hid_dims))
        )

    def init_right_hidden(self):
        '''
        initialize right-chain-lstm's hidden state and memory cell state\n
         '''
        return (
            Variable(torch.zeros(1, 1, self.r_hid_dims)),
            Variable(torch.zeros(1, 1, self.r_hid_dims))
        )

    def bottom_up(self, graph=None):

        '''
        bottom up direction transform computation
        @Parameter \n
        graph : SemanticGraph class object\n
        '''
        # iterator stack for DFS
        stack = []

        # push root node iterator into stack
        root_ite = sstrut.SemanticGraphIterator(graph.root, graph)
        stack.append(root_ite)

        # DFS on the parse tree
        while len(stack) is not 0:
            ite = stack[len(stack) - 1]

            if ite.isLeaf():
                # leaf node with specific transformation
                # do something
                o_list = ite.getOutgoingEdges()
                p_list = ite.getIncomingEdges()
                p = ite.getParent()
                c = ite.getChildren()
                print(ite.node.text)
                stack.pop()

            else:
                if ite.allChildrenChecked():
                    # if all children are computed then transform parent node with children
                    # do something
                    o_list = ite.getOutgoingEdges()
                    p_list = ite.getIncomingEdges()
                    p = ite.getParent()
                    c = ite.getChildren()
                    print(ite.node.text)
                    stack.pop()

                else:
                    # else traverse parent node's next child node
                    stack.append(ite.next())

    def getEmbeddings(self, graph):
        pass

    def represent(self):
        pass

    def bi_lstm_computing(self):
        pass

    def forward(self, graph=None):

        '''
        override nn.Module's forward method to implement bottom up direction
        hierarchical information propagation
        '''

        if graph is None:
            print("forward pass")
        else:
            # self.getEmbeddings(graph)
            # self.bi_lstm_computing(graph)
            self.bottom_up(graph)
