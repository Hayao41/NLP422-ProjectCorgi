import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from semantic.SemanticStructure import SemanticGraphIterator as siterator
from queue import Queue
import gc

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
        self.use_cuda = options.use_cuda

    def bottom_up(self, graph):
        
        '''
        bottom-up direction tranformation computation on parse tree(DFS)\n
        @Parameter \n
        graph : SemanticGraph class object\n
        '''
        # iterator ite_stack for DFS
        ite_stack = []

        # push root node iterator into ite_stack
        graph.root.pushed = True
        root_ite = siterator(graph.root, graph)
        ite_stack.append(root_ite)

        # DFS on parse tree
        while len(ite_stack) is not 0:
            ite = ite_stack[-1]

            if ite.allChildrenChecked():
                # if all children have checked (leaf node has no children
                # so that it's all children have been checked by default)
                self.bu_transform(ite)
                ite_stack.pop()

            else:
                ite_stack.append(next(ite))

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

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def forward(self):
        raise NotImplementedError
        

class HierarchicalTreeLSTMs(TreeStructureNetwork):
    
    ''' 
    Model implements HierarchicalTreeLSTMs(Kiperwasser, Yoav., 2016, https://arxiv.org/abs/1603.00375)\n
    This class contains three lstm unit for supporting hierarchical lstms for encoding a given
    dependency parse tree.\n
    @Attribute:\n
    enc_linear: non-linear trans for combining left chain final state and right chain final state\n
    l_lstm: LSTMs unit for computing left children vector\n
    r_lstm: LSTMs unit for computing right children vector\n
    '''
    
    def __init__(self, options):
        super(HierarchicalTreeLSTMs, self).__init__(options)
        
        self.lstm_hid_dims = options.lstm_hid_dims
        self.rel_emb_dims = options.rel_emb_dims

        # chain initial state params
        self.total_layers = options.chain_num_layers * options.chain_direction
        self.single_pass_dims = options.chain_hid_dims // options.chain_direction

        # Encoding children concatenated vector linear layer
        #
        # enc(t) = g(W( el(t) con er(t) ) + b)
        #   where con is the concatenate operator
        #
        # if arc_relation representation is needed, the encoding function is changed to
        # enc(t) = g(W( el(t) con er(t) con v(rel)) + b)
        #   where con is the concatenate operator, 
        #   v(rel) is relation between current word and head word
        #
        # specially for leaf
        # enc(leaf) = g(W( el(leaf) con er(leaf) ) + b)
        #
        # @Dimension : l_hid_dims + r_hid_dims -> lstm_hid_dims
        #   or l_hid_dims + r_hid_dims + rel_emb_dims -> lstm_hid_dims
        self.enc_linear = nn.Linear(
            options.chain_hid_dims * 2 + options.rel_emb_dims,
            options.lstm_hid_dims
        )

        # LSTMs unit computing left children vector
        #
        # el(t) = RNNl(vi(t), enc(t.l1), enc(t.l2), ... , enc(t.lk))
        #
        # specially for leaf
        # el(leaf) = RNNl(vi(leaf))
        #
        # @Dimension : lstm_hid_dims -> l_hid_dims
        self.l_lstm = nn.LSTM(
            options.lstm_hid_dims,
            options.chain_hid_dims // options.chain_direction,
            num_layers=options.chain_num_layers,
            dropout=options.dropout,
            bidirectional=options.use_bi_chain
        )

        # LSTMs unit computing right children vector
        #
        # er(t) = RNNr(vi(t), enc(t.r1), enc(t.r2), ... , enc(t.rk))
        #
        # specially for leaf
        # er(leaf) = RNNr(vi(leaf))
        #
        # @Dimensions : lstm_hid_dims -> r_hid_dims
        self.r_lstm = nn.LSTM(
            options.lstm_hid_dims,
            options.chain_hid_dims // options.chain_direction,
            num_layers=options.chain_num_layers,
            dropout=options.dropout,
            bidirectional=options.use_bi_chain
        )

        if options.xavier:
            self.init_weights

    def init_weights(self):
        nn.init.xavier_normal(self.enc_linear.weight)

    def bu_transform(self, iterator):
        
        ''' 
        bottom up direction tree encdoing transformation\n
        @Trans enc(t) = g(W( el(t) con er(t) ) + b)\n
        @Trans el(t) = RNNl(vi(t), enc(t.l1), enc(t.l2), ... , enc(t.lk))\n
        @Trans er(t) = RNNr(vi(t), enc(t.r1), enc(t.r2), ... , enc(t.rk))\n
        '''

        if(iterator.node.text=="http:"):
            print(iterator.node)

        # left chain last hidden state
        left_state = self.left_chain(iterator)[-1].view(1, -1)

        # right chain last hidden state
        right_state = self.right_chain(iterator)[-1].view(1, -1)

        # incom-relation embedding
        if self.rel_emb_dims is not 0:
            incom_rel = list(iterator.queryIncomRelation())[0].rel_vec.view(1, -1)
        else:
            incom_rel = None

        # concatenate non-linear trans
        hidden_vector = self.combination(left_state, right_state, incom_rel=incom_rel)

        # set context vector(as memory to next recursive stage)
        iterator.node.context_vec = hidden_vector

    def tp_transform(self, iterator):
        
        # here can be dynamic routing block

        pass

    def left_chain(self, iterator):
        
        ''' 
        left children chain transformation from head word to most left word
        '''
        
        # stack vectors from head node to most left node
        left_hiddens = [iterator.node.context_vec.view(1, -1)] + list(iterator.left_hiddens())
        left_chain = torch.cat((left_hiddens), 0)

        return self.chain_transform(left_chain, self.l_lstm)

    def right_chain(self, iterator):
        
        ''' 
        right children chain transformation from head word to most right word
        '''

        right_hiddens = [iterator.node.context_vec.view(1, -1)] + list(iterator.right_hiddens())
        right_chain = torch.cat((right_hiddens), 0)

        return self.chain_transform(right_chain, self.r_lstm)

    def combination(self, left_state, right_state, incom_rel=None):
        
        ''' 
        head word encoding : left and right chain combination transformation\n
        @Trans enc(t) = g(W( el(t) con er(t) ) + b) where g is tanh
        @ExdTrans enc(t) = g(W( el(t) con er(t) con v(rel)) + b) where g is tanh
        '''
        if self.rel_emb_dims is not 0:
            con_vec = torch.cat((left_state, right_state, incom_rel), -1)
        else:
            con_vec = torch.cat((left_state, right_state), -1)
        out = self.enc_linear(con_vec)
        hidden_vector = F.tanh(out)
        return hidden_vector

    def chain_transform(self, chain, lstm):
        
        ''' 
        children chain transformation\n
        @Trans e(t) = RNN(vi(t), enc(t.c1), enc(t.c2), ... , enc(t.ck)) 
        '''

        def init_hidden():
            '''
            initialize two_branch_chain_lstm's hidden state and memory cell state\n
            '''

            if self.use_cuda:
                return (
                    Variable(torch.zeros(
                        self.total_layers, 
                        1, 
                        self.single_pass_dims)
                    ).cuda(),
                    Variable(torch.zeros(
                        self.total_layers, 
                        1, 
                        self.single_pass_dims)
                    ).cuda()
                )
            else:
                return (
                    Variable(torch.zeros(
                        self.total_layers, 
                        1, 
                        self.single_pass_dims)
                    ),
                    Variable(torch.zeros(
                        self.total_layers, 
                        1, 
                        self.single_pass_dims)
                    )
                )
        
        hidden_states = init_hidden()
        # gc.collect()

        out, hidden_states = lstm(
            chain.view(-1, 1, self.lstm_hid_dims),
            hidden_states
        )
        return out

    def forward(self, graph):

        assert graph is not None, "[Error] Tree model's input graph is None type!"

        print("Training on {}".format(graph.sid))

        self.bottom_up(graph)


class DynamicRecursiveNetwork(TreeStructureNetwork):
    
    def __init__(self, options, att_layer, dy_rout):
        super(DynamicRecursiveNetwork, self).__init__(options)
        self.att_layer = att_layer
        self.dy_rout = dy_rout

    def bu_transform(self, iterator):
        self.att_layer()

    def tp_transform(self, iterator):
        self.dy_rout()

    def forward(self, graph):
        
        assert graph is not None, "[Error] Tree model's input graph is None type!"

        self.bottom_up(graph)

        self.top_down(graph)
