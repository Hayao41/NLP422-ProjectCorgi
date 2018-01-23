
import torch
import torch.nn as nn
from collections import namedtuple


class ClauseDetector(nn.Module):
    ''' 
    ClauseDetector framework encapsulates nn Modules to modeling sentence and 
    semantic graph to capture sentence semantic and structure information\n
    @Modules:\n
    encoder : ContextEncoder\n
    tree : TreeNeuralNetwork\n
    tree_embed : Tree structure informatio embedding\n 
    clf : MLP(Multi layer perception)\n
    '''
    def __init__(self, options, encoder, tree_embed, tree, classifier):
        super(ClauseDetector, self).__init__()
        self.use_cuda = options.use_cuda
        self.encoder = encoder
        self.tree_embed = tree_embed
        self.tree = tree
        self.clf = classifier

    def switch2gpu(self):
        self.use_cuda = True
        self.encoder.switch2gpu()
        self.tree_embed.switch2gpu()
        self.tree.switch2gpu()
        self.clf.switch2gpu()

    def switch2cpu(self):
        self.use_cuda = False
        self.encoder.switch2cpu()
        self.tree_embed.switch2cpu()
        self.tree.switch2cpu()
        self.clf.switch2cpu()

    def mapSequence2Graph(self, context_vecs, batch_graph):
        
        ''' 
        set context encoder's encoded vectors onto semantic graph 
        and clip padding word at the same time 
        '''

        batch_size = len(batch_graph)

        assert context_vecs.size()[0] == batch_size, "[Error] context vectors' batch size does not match graphs'!"

        for inst in range(batch_size):
            sequence = context_vecs[inst]
            graph = batch_graph[inst]
            for idx in range(len(graph.indexedWords)):
                graph.indexedWords[idx].context_vec = sequence[idx]

    def mapGraph2Sequence(self, batch_graph):
        pass


    def makeOuputTuple(self, preds):
        ''' 
        makes output namedtuple cuz ResNN cont be batched accelerating 
        @return\n
        namedtuple('OutputTuple', ['outputs', 'preds'])\n
        outputs : batch predictions tensor matrix
        preds : batch predictions list
        '''
        outputs = torch.cat((preds), 0)
        OutputTuple = namedtuple('OutputTuple', ['outputs', 'preds'])
        return OutputTuple(outputs=outputs, preds=preds)

    def forward(self, batch_data):
        
        self.zero_grad()
        
        batch_sequence, batch_graph = batch_data

        assert len(batch_sequence) == len(batch_graph), "[Error] sequences' batch size does not match graphs'!"

        # sequence context encoings
        context_vecs = self.encoder(batch_sequence)

        # map sequence context vectors onto tree(resursive model is hard to batch compute)
        self.mapSequence2Graph(context_vecs, batch_graph)

        # tree structure information embedding(arc relation, relative position etc.)
        self.tree_embed(batch_graph)

        preds = []

        # tree encoding and classify
        for graph in batch_graph:
            self.tree(graph)
            preds.append(self.clf(graph.getContextVecs()))

        return self.makeOuputTuple(preds)
