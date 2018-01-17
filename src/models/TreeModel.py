import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from semantic.SemanticStructure import SemanticGraphIterator as siterator
from queue import Queue


class TreeStructureNetwork(nn.Module):
    
    ''' 
    base class of tree structure neural networks which can propagate information 
    recursively with two directions(bottom up and top down). If you want to implement your tree model, 
    extend this class and implement two type transform method

    >>> class Mytree(TreeStructureNetwork):
    >>>     def bu_transform(iterator):
    >>>         "your implementation"
    >>>     def tp_transform(iterator):
    >>>         "your implementation"
    '''

    def __init__(self, options):
        super(TreeStructureNetwork, self).__init__()
        self.options = options

    def bottom_up(self, graph):
        
        '''
        bottom-up direction tranformation computation on parse tree(DFS)\n
        @Parameter \n
        graph : SemanticGraph class object\n
        '''
        # iterator ite_stack for DFS
        ite_stack = []

        # push root node iterator into ite_stack
        root_ite = siterator(graph.root, graph)
        ite_stack.append(root_ite)

        # DFS on parse tree
        while len(ite_stack) is not 0:
            ite = ite_stack[len(ite_stack) - 1]

            if ite.allChildrenChecked():
                # if all children have checked (leaf node has no children
                # so that it's all children have been checked by default)
                self.bu_transform(ite)
                ite_stack.pop()

            else:
                ite_stack.append(ite.next())

    def top_down(self, graph):
        
        '''
        top down direction transformation computation on parse tree(BFS)\n
        @Parameter \n
        graph : SemanticGraph class object\n
        '''
    
        # iterator queue for BFS
        ite_queue = Queue()

        # push root node iterator into queue
        root_ite = siterator(graph.root, graph)
        ite_queue.put(root_ite)

        # BFS on parse tree
        while not ite_queue.empty():
            ite = ite_queue.get()

            # do something
            self.tp_transform(ite)

            # push current node's children into queue
            for child in ite.children():
                ite_queue.put(child)

    def bu_transform(self, iterator):
    
        ''' 
        base bottom up transformation function for recursive trans\n
        @Param : semantic structure iterator
        '''
        raise NotImplementedError

    def tp_transform(self, iterator):
        
        ''' 
        base top down transformation function for recursive trans\n
        @Param : semantic structure iterator
        '''
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
        

class HierarchicalTreeLSTMs(TreeStructureNetwork):
    
    ''' 
    Model implements HierarchicalTreeLSTMs(Yoav et al., 2016, https://arxiv.org/abs/1603.00375)\n
    This class contains three lstm unit for supporting hierarchical lstms for encoding a given
    dependency parse tree.\n
    '''
    
    def __init__(self, options):
        super(HierarchicalTreeLSTMs, self).__init__(options)

        self.hidden_states = (
            Variable(torch.zeros(1, 1, self.options.l_hid_dims)),
            Variable(torch.zeros(1, 1, self.options.l_hid_dims))
        )

        # Encoding children concatenated vector linear layer
        #
        # enc(t) = g(W( el(t) con er(t) ) + b)
        #   where con is the concatenate operator
        #
        # if arc_relation representation is needed, the encoding function is changed to
        # enc(t) = g(W( el(t) con er(t) con rel_emb) + b)
        #   where con is the concatenate operator, 
        #   rel_emb is relation between current word and head word
        #
        # specially for leaf
        # enc(leaf) = g(W( el(leaf) con er(leaf) ) + b)
        #
        # @Dimension : l_hid_dims + r_hid_dims -> bi_hid_dims
        #   or l_hid_dims + r_hid_dims + rel_emb_dims -> bi_hid_dims
        self.enc_linear = nn.Linear(
            self.options.l_hid_dims + self.options.r_hid_dims + self.options.rel_emb_dims,
            self.options.bi_hid_dims,
            bias=True
        )

        # LSTM unit computing left children vector
        #
        # el(t) = RNNl(vi(t), enc(t.l1), enc(t.l2), ... , enc(t.lk))
        #
        # specially for leaf
        # el(leaf) = RNNl(vi(leaf))
        #
        # @Dimension : bi_hid_dims -> l_hid_dims
        self.l_lstm = nn.LSTM(
            self.options.bi_hid_dims,
            self.options.l_hid_dims
        )

        # LSTM unit computing right children vector
        #
        # er(t) = RNNr(vi(t), enc(t.r1), enc(t.r2), ... , enc(t.rk))
        #
        # specially for leaf
        # er(leaf) = RNNr(vi(leaf))
        #
        # @Dimensions : bi_hid_dims -> r_hid_dims
        self.r_lstm = nn.LSTM(
            self.options.bi_hid_dims,
            self.options.r_hid_dims
        )

        if options.xavier:
            self.init_weights

    def init_hidden(self):
        '''
        initialize two_branch_chain_lstm's hidden state and memory cell state\n
        '''
        self.hidden_states = (
            Variable(torch.zeros(1, 1, self.options.l_hid_dims)),
            Variable(torch.zeros(1, 1, self.options.l_hid_dims))
        )

    def init_weights(self):
        nn.init.xavier_normal(self.enc_linear.weight)

    def bu_transform(self, iterator):
        
        ''' 
        bottom up direction tree encdoing transformation\n
        @Trans enc(t) = g(W( el(t) con er(t) ) + b)\n
        @Trans el(t) = RNNl(vi(t), enc(t.l1), enc(t.l2), ... , enc(t.lk))\n
        @Trans er(t) = RNNr(vi(t), enc(t.r1), enc(t.r2), ... , enc(t.rk))\n
        '''

        # left chain last hidden state
        left_state = self.left_chain(iterator)[-1].view(-1, self.options.l_hid_dims)

        # right chain last hidden state
        right_state = self.right_chain(iterator)[-1].view(-1, self.options.r_hid_dims)
        
        # concatenate non-linear trans
        hidden_vector = self.combination(left_state, right_state)

        # set context vector(as memory to next recursive stage)
        iterator.node.context_vec = hidden_vector

    def tp_transform(self, iterator):
        # print(iterator.node.text)
        pass

    def left_chain(self, iterator):
        
        ''' 
        left children chain transformation from head word to most left word
        '''
        
        left_chain = iterator.node.context_vec

        for left_hidden in iterator.left_hiddens():
            left_chain = torch.cat((left_hidden.view(1, -1), left_chain.view(1, -1)))

        return self.chain_transform(left_chain, self.l_lstm)

    def right_chain(self, iterator):
        
        ''' 
        right children chain transformation from head word to most right word
        '''
        
        right_chain = iterator.node.context_vec

        for right_hidden in iterator.right_hiddens():
            right_chain = torch.cat((right_chain.view(1, -1), right_hidden.view(1, -1)), 1)

        return self.chain_transform(right_chain, self.r_lstm)

    def combination(self, left_state, right_state):
        
        ''' 
        head word encoding : left and right chain combination transformation\n
        @Trans enc(t) = g(W( el(t) con er(t) ) + b) where g is tanh
        '''

        con_vec = torch.cat((left_state, right_state), 1)
        out = self.enc_linear(con_vec)
        hidden_vector = F.tanh(out)
        return hidden_vector

    def chain_transform(self, chain, lstm):
        
        ''' 
        children chain transformation\n
        @Trans e(t) = RNN(vi(t), enc(t.c1), enc(t.c2), ... , enc(t.ck)) 
        '''
        
        self.init_hidden()

        out, self.hidden_states = lstm(
            chain.view(-1, 1, self.options.bi_hid_dims),
            self.hidden_states
        )
        return out

    def forward(self, graph=None):

        if graph is None:
            print("forward pass")
        else:
            self.bottom_up(graph)
            self.top_down(graph)


class DynamicRecursiveNetwork(TreeStructureNetwork):
    
    def __init__(self, options, att_layer, dy_rout):
        super(DynamicRecursiveNetwork, self).__init__(options)
        self.att_layer = att_layer
        self.dy_rout = dy_rout

    def bu_transform(self, iterator):
        self.att_layer()

    def tp_transform(self, iterator):
        self.dy_rout()

    def forward(self, graph=None):
        if graph is None:
            print("forward pass")
        else:
            self.bottom_up(graph)
            self.top_down(graph)

class AttentionUnit(nn.Module):
    
    def __init__(self):
        pass

class DynamicRoutingUnit(nn.Module):
    
    def __init__(self):
        pass


class TestModel(nn.Module):

    ''' 
    test model that implements detector encapsulate embedding layer, context encoder, tree encoder
    for every layer testing
    '''
    
    def __init__(self, embed, tree_model, encoder):
        super(TestModel, self).__init__()
        self.embed = embed
        self.tree_model = tree_model
        self.encoder = encoder
        self.linear = nn.Linear(tree_model.options.bi_hid_dims, 2, bias=True)

        nn.init.xavier_normal(self.linear.weight)

    def zero_grad(self):
        self.embed.zero_grad()
        self.encoder.zero_grad()
        self.tree_model.zero_grad()
        self.linear.zero_grad()

    def classify(self, graph):
        result = []
        for word in graph.indexedWords:
            out = self.linear(word.context_vec.view(-1, self.tree_model.options.bi_hid_dims))
            out = F.log_softmax(out, 1)
            result.append(out)
        
        for index in range(len(result) - 1):
            result[index + 1] = torch.cat((result[index], result[index + 1]))

        return result[-1]

    def setContextVecotr2Graph(self, batch_sequence, batch_graph):
        pass

    def forward(self, batch_data):
        
        batch_sequence, batch_graph = batch_data

        # sequence encoding
        embedded_sequence = self.embed(batch_sequence)
        context_vectors = self.encoder(embedded_sequence)

        # graph encoding
        self.setContextVecotr2Graph(context_vectors, batch_graph)
        self.tree_model(batch_graph)

        # calssify
        out = self.classify(batch_graph)
        
        return out
