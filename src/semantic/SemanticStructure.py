''' 
This file is semantic structure definition
 '''
import torch
from collections import namedtuple


class SemanticGraph(object):

    ''' 
    Semantic graph stores the dependency tree structure\n
    @Attributes :
    root : root node of tree structure semantic graph\n
    edgesOutgoing : outgoing edge map table\n
    edgesIncoming : incoming edge map table\n
    indexedWords : sentence order node list
    '''

    def __init__(self, 
                sid='S#',
                root=None,
                edgesOutgoing=None,
                edgesIncoming=None,
                indexedWords=None,
                hasCycle=False
                ):
        super(SemanticGraph, self).__init__()
        self.sid = sid
        self.root = root
        self.edgesOutgoing = edgesOutgoing
        self.edgesIncoming = edgesIncoming
        self.indexedWords = indexedWords
        self.hasCycle = hasCycle

    def getLabels(self):
        
        for word in self.indexedWords:
            yield word.label

    def outcomingEdges(self):
        
        for source in self.edgesOutgoing:
            edges = self.edgesOutgoing[source]
            for edge in edges:
                yield edge

    def incomingEdges(self):
        
        for target in self.edgesIncoming:
            edges = self.edgesIncoming[target]
            for edge in edges:
                yield edge

    def getNodePositionIdxs(self):
        
        for word in self.indexedWords:
            yield word.rp_idx

    def setNodePositionEmbeddings(self, position_embeddings):
        
        for index in range(len(self.indexedWords)):
            self.indexedWords[index].rp_vec = position_embeddings[index]


    def getArcRelationIdxs(self):
    
        for edge in self.incomingEdges():
            yield edge.rel_idx

    def setArcRelationEmbeddings(self, rel_embeddings):
        
        index = 0
        for edge in self.incomingEdges():
            edge.rel_vec = rel_embeddings[index]
            index = index + 1
    
    def getContextVecs(self):
        
        vec_list = [word.context_vec for word in self.indexedWords]
        context_vecs = torch.cat((vec_list), 0)
        return context_vecs

    def clean_up(self):
        
        ''' 
        clean up encoded vector of node from semantic graph. if you do not clean up, 
        it may cause memory leak during training stage becuz it, Variable object, contains
        full computation graph
        '''
        
        for word in self.indexedWords:
            del word.context_vec

    def __len__(self):
        return len(self.indexedWords)

    def __str__(self):

        graphStr = ""
        graphStr += self.sid + "\n"
        for word in self.indexedWords:
            graphStr += str(word) + "\n"
        return graphStr
    

class SemanticGraphNode(object):
    
    def __init__(self,
                text=None,
                pos=None,
                sentence_index=None,
                parent_index=None,
                word_idx=None,
                pos_idx=None,
                rp_idx=None,
                rp_vec=None,
                context_vec=None,
                atten_prob=0.,
                coupling_prob=0.,
                label=0,
                isLeaf=False
    ):
        super(SemanticGraphNode, self).__init__()
        self.text = text
        self.pos = pos
        assert isinstance(sentence_index, int), "[Error] sentence index must be <int> data type but got {type}".format(
            type=type(sentence_index)
        )
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.word_idx = word_idx
        self.pos_idx = pos_idx
        self.rp_idx = rp_idx
        self.rp_vec = rp_vec
        self.context_vec = context_vec
        self.label = label
        self.atten_prob = atten_prob
        self.coupling_prob = coupling_prob
        self.isLeaf = isLeaf

    def __str__(self):
        
        label = ""
        if self.label != 0:
            label = "CLASUE_SPLIT"
        else:
            label = "NO_ACTION"

        if self.isLeaf:
            leaf = "True"
        else:
            leaf = "False"

        return "Text:[{text}], POS:[{pos}], Index:[{index}], Leaf[{leaf}], Label:[{label}]".format(
            text=self.text,
            pos=self.pos,
            index=self.sentence_index,
            leaf=leaf,
            label=label
        )


class SemanticGraphEdge(object):
    
    def __init__(self,
                source=None,
                target=None,
                relation=None,
                rel_idx=None,
                rel_vec=None
                ):
        super(SemanticGraphEdge, self).__init__()
        self.source = source
        self.target = target
        self.relation = relation
        self.rel_idx = rel_idx
        self.rel_vec = rel_vec

    def __str__(self):
        return "(Source)[{}] -> (Target)[{}]".format(
            str(self.source),
            str(self.target)
        )


class SemanticGraphIterator(object):

    '''
    iterator for traversing semantic graph structure\n
    @Iterable\n
    returns current node's next child
    '''

    def __init__(self, node, graph):
        super(SemanticGraphIterator, self).__init__()
        self.node = node
        self.graph = graph

        # list for recording unchecked children
        self.c_list = list(self.children())

    def getOutgoingEdges(self):
        if self.node in self.graph.edgesOutgoing:
            for edge in self.graph.edgesOutgoing[self.node]:
                yield edge
        else:
            # if there are no outgoing edges starts with current node, stop iteration
            raise StopIteration

    def getIncomingEdges(self):
        if self.node in self.graph.edgesIncoming:
            for edge in self.graph.edgesIncoming[self.node]:
                yield edge
        else:
            # if there are no incoming edges ends with current node, stop iteration
            raise StopIteration

    def children(self):
        
        for edge in self.getOutgoingEdges():
            yield SemanticGraphIterator(edge.target, self.graph)

    def parents(self):
        
        for edge in self.getIncomingEdges():
            yield SemanticGraphIterator(edge.source, self.graph)

    def queryIncomRelation(self):
        
        ''' get current node's incoming arc relation list  '''

        IncomRelation = namedtuple('IncomRelation', ['relation', 'rel_idx', 'rel_vec'])
        
        for edge in self.getIncomingEdges():
            yield IncomRelation(
                relation=edge.relation,
                rel_idx=edge.rel_idx,
                rel_vec=edge.rel_vec
            )

    def setAttentionProbs(self, weights):
        
        for index, edge in enumerate(self.getOutgoingEdges()):
            edge.target.atten_prob = weights[index].cpu().data[0]

    def setCouplingProbs(self, weights):
        
        for index, edge in enumerate(self.getOutgoingEdges()):
            edge.target.coupling_prob = weights[index].cpu().data[0]

    def children_hiddens(self):
        
        ''' get current node's children hidden states '''
        
        for edge in self.getOutgoingEdges():
            target = edge.target
            yield target.context_vec

    def left_hiddens(self):
        
        ''' 
        get current node's left side words matrix form current to 
        most left(sentence index ordered by DES)
        '''
        
        targets = []
        
        for edge in self.getOutgoingEdges():
            target = edge.target
            if target.sentence_index < self.node.sentence_index:
                targets.append(target)
        
        if len(targets) is not 0:
            # sort right target nodes by sentence index DES
            targets.sort(reverse=True, key=lambda target: target.sentence_index)

            for target in targets:
                yield target.context_vec

        else:
            raise StopIteration
    
    def right_hiddens(self):
        
        ''' 
        get current node's right side words matrix form current to 
        most right(sentence index ordered by ASC)
        '''
        
        targets = []
        
        for edge in self.getOutgoingEdges():
            target = edge.target
            if target.sentence_index > self.node.sentence_index:
                targets.append(target)

        if len(targets) is not 0:
            # sort right target nodes by sentence index ASC
            targets.sort(key=lambda target: target.sentence_index)

            for target in targets:
                yield target.context_vec
            
        else:
            raise StopIteration

    def isLeaf(self):
        
        if self.node.isLeaf is not None:
            return self.node.isLeaf
        else:
            return len(list(self.children())) == 0

    def allChildrenChecked(self):
        
        # if children record list is empty, current node's children all have checked
        return len(self.c_list) == 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        next_ite = self.c_list[0]
        self.c_list.remove(next_ite)
        return next_ite        


class Sequence(object):
    
    ''' 
    sequential data structure for sentence, containing words index sequence tenor and pos index sequence tensor\n
    @Attribute\n
    words_tensor: word index sequence 1d tensor(wrapped by Variable, if train_batch_size > 1 it should be 2d)
    pos_tensor: pos index sequence 1d tensor(wrapped by Variable, if train_batch_size > 1 it should be 2d)
    train_batch_size: mini batch size, for supporting mini batch, sequences can be stacked to a batched 2d tensor
    '''
    
    def __init__(self, words_tensor=None, pos_tensor=None, batch_size=1, use_cuda=False):
        
        super(Sequence, self).__init__()
        self.words_tensor = words_tensor
        self.pos_tensor = pos_tensor
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def __len__(self):
        return self.batch_size

    def switch2gpu(self):

        self.use_cuda = True
        
        if self.words_tensor is not None:
            self.words_tensor = self.words_tensor.cuda()
        if self.pos_tensor is not None:
            self.pos_tensor = self.pos_tensor.cuda()

    def switch2cpu(self):

        self.use_cuda = False
        
        if self.words_tensor is not None:
            self.words_tensor = self.words_tensor.cpu()
        if self.pos_tensor is not None:
            self.pos_tensor = self.pos_tensor.cpu()
