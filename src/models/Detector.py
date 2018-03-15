
import torch
import torch.nn as nn
from collections import namedtuple


class ClauseDetector(nn.Module):
    ''' 
    ClauseDetector framework encapsulates nn Modules to modeling sentence and 
    semantic graph to capture sentence semantic and structure information\n
    @Modules:\n
    context_encoder : ContextEncoder\n
    tree_embed : Tree structure information embedding\n 
    tree_encoder : TreeModel\n
    clf : MLP(Multi layer perception)\n
    '''
    def __init__(self, options, context_encoder, tree_embed, tree_encoder, classifier):
        super(ClauseDetector, self).__init__()
        self.use_cuda = options.use_cuda
        self.context_encoder = context_encoder
        self.tree_embed = tree_embed
        self.tree_encoder = tree_encoder
        self.clf = classifier

    def switch2gpu(self):
        self.use_cuda = True
        self.context_encoder.switch2gpu()
        self.tree_embed.switch2gpu()
        self.tree_encoder.switch2gpu()
        self.clf.switch2gpu()

    def switch2cpu(self):
        self.use_cuda = False
        self.context_encoder.switch2cpu()
        self.tree_embed.switch2cpu()
        self.tree_encoder.switch2cpu()
        self.clf.switch2cpu()

    def mapSequence2Graph(self, context_vecs, batch_graph):
        
        ''' 
        set context context_encoder's encoded vectors onto semantic graph 
        and clip padding word at the same time 
        '''

        batch_size = len(batch_graph)

        assert context_vecs.size()[0] == batch_size, "[Error] context vectors' batch size does not match graphs'!"

        for inst in range(batch_size):
            sequence = context_vecs[inst]
            graph = batch_graph[inst]
            for idx in range(len(graph.indexedWords)):
                graph.indexedWords[idx].context_vec = sequence[idx]

    def makeOuputTuple(self, preds):
        
        ''' 
        make output namedtuple cuz ResNN can't be batched accelerating(waiting for high performance solution)\n
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

        # sequence context encoding
        context_vecs = self.context_encoder(batch_sequence)

        # map sequence context vectors onto tree(resursive model is hard to batch accelerating)
        self.mapSequence2Graph(context_vecs, batch_graph)

        # tree structure information embedding(arc relation, relative position etc.)
        self.tree_embed(batch_graph)

        preds = []

        # tree encoding and classify
        for graph in batch_graph:
            self.tree_encoder(graph)
            preds.append(self.clf(graph.getContextVecs()))

        return self.makeOuputTuple(preds)
