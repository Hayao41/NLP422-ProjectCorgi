''' 
This file is semantic structure definition
 '''
import torch
from torch.autograd import Variable


class SemanticGraph(object):

    ''' 
    Semantic graph stores the dependency tree structure\n
    @Attributes :
    root : the root node of tree structure semantic graph\n
    outgoing_edges : outgoing edge map table\n
    incoming_edges : incoming edge map table\n
    '''

    def __init__(self, 
                root=None,
                outgoing_edges=None,
                incoming_edges=None,
                sentence=None
                ):
        super(SemanticGraph, self).__init__()
        self.root = root
        self.outgoing_edges = outgoing_edges
        self.incoming_edges = incoming_edges
        self.sentence = sentence

    def getWordIdxs(self):
        return self.sentence.getWordIdxs()

    def getPOSIdxs(self):
        return self.sentence.getPOSIdxs()

    def getLabels(self):
        return self.sentence.getLabels()

    def getArcRelationIdxs(self):
        
        relIdxs = []
        edges = self.getOutgoingEdges()
        for edge in edges:
            relIdxs.append(edge.rel_idx)
        return relIdxs

    def getOutgoingEdges(self):
        
        edges = []
        for source in self.outgoing_edges:
            edge_list = self.outgoing_edges[source]
            for edge in edge_list:
                edges.append(edge)
        return edges

    def getContextVectors(self):

        context_vecotrs = []
        for word in self.indexedWords:
            context_vecotrs.append(word.context_vec)
        return context_vecotrs

    def setWordEmbeddings(self, word_embeddings):
        
        self.word_embeddings = word_embeddings
        for index in range(len(word_embeddings)):
            self.indexedWords[index].word_vec = word_embeddings[index]

    def setPOSEmbeddings(self, pos_embeddings):
        
        self.pos_embeddings = pos_embeddings
        for index in range(len(pos_embeddings)):
            self.indexedWords[index].pos_vec = pos_embeddings[index]

    def setArcRelationEmbeddings(self, rel_embeddings):
        
        edges = self.getOutgoingEdges()
        for index in range(len(rel_embeddings)):
            edges[index].rel_vec = rel_embeddings[index]

    def setContextVector(self, context_vectors):
        
        for index in range(len(self.indexedWords)):
            self.indexedWords[index].context_vec = context_vectors[index]


class sentence(object):
    
    def __init__(self):
        
        self.indexedWords = []

    def getWordIdxs(self):
        word_idxs = []
        for word in self.indexedWords:
            word_idxs.append(word.word_idx)
        
        return Variable(torch.LongTensor(word_idxs))

    def getPOSIdxs(self):
        pos_idxs = []
        for word in self.indexedWords:
            pos_idxs.append(word.pos_idx)
        
        return Variable(torch.LongTensor(pos_idxs))

    def getLabels(self):
        word_labels = []
        for word in self.indexedWords:
            word_labels.append()

    def getWordEmbeddings(self):
        pass

    def getPosEmbeddings(self):
        pass

    def append(self, object):
        self.indexedWords.append(object)
    

class SemanticGraphNode(object):
    
    def __init__(self,
                text=None,
                pos=None,
                sentence_index=None,
                parent_index=None,
                word_idx=None,
                pos_idx=None,
                rp_idx=None,
                label=0,
                rp_vec=None,
                context_vec=None,
                isLeaf=False
    ):
        super(SemanticGraphNode, self)
        self.text = text
        self.pos = pos
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.word_idx = word_idx
        self.pos_idx = pos_idx
        self.rp_idx = rp_idx
        self.label = label
        self.rp_vec = rp_vec
        self.context_vec = context_vec
        self.isLeaf = isLeaf


class SemanticGraphEdge(object):
    
    def __init__(self,
                source=None,
                target=None,
                relation=None,
                rel_idx=None,
                rel_vec=None
                ):
        super(SemanticGraphEdge, self)
        self.source = source
        self.target = target
        self.relation = relation
        self.rel_idx = rel_idx
        self.rel_vec = rel_vec


class SemanticGraphIterator(object):

    '''
    iterator for traversing semantic graph structure
    '''

    def __init__(self, node, graph):
        super(SemanticGraphIterator, self).__init__()
        self.node = node
        self.graph = graph

        # list for recording unchecked children node
        self.c_list = self.getChildrenList()

    def getChildrenList(self):
        children_list = []
        for edge in self.getOutgoingEdges():
            children_list.append(SemanticGraphIterator(edge.target, self.graph))
        return children_list

    def getParentList(self):
        parent_list = []
        for edge in self.getIncomingEdges():
            parent_list.append(SemanticGraphIterator(edge.source, self.graph))
        return parent_list

    def getOutgoingEdges(self):
        if self.node in self.graph.outgoing_edges:
            for edge in self.graph.outgoing_edges[self.node]:
                yield edge
        else:
            # if there are no outgoing edges starts with current node return a empty list
            temp = []
            return temp

    def getIncomingEdges(self):
        if self.node in self.graph.incoming_edges:
            for edge in self.graph.incoming_edges[self.node]:
                yield edge
        else:
            temp = []
            return temp

    def children(self):
        
        for edge in self.getOutgoingEdges():
            yield SemanticGraphIterator(edge.target, self.graph)

    def parents(self):
        
        for edge in self.getIncomingEdges():
            yield SemanticGraphIterator(edge.source, self.graph)

    def left_hiddens(self):
        
        for edge in self.getOutgoingEdges():
            if edge.target.sentence_index < self.node.sentence_index:
                yield edge.target.context_vec
    
    def right_hiddens(self):
        
        for edge in self.getOutgoingEdges():
            if edge.target.sentence_index > self.node.sentence_index:
                yield edge.target.context_vec

    def isLeaf(self):
        
        if self.node.isLeaf is not None:
            return self.node.isLeaf
        else:
            return len(self.getChildren()) == 0

    def allChildrenChecked(self):
        
        # if children record list is empty, current node's children all have checked
        return len(self.c_list) == 0

    def next(self):
        next_ite = self.c_list[0]
        self.c_list.remove(next_ite)
        return next_ite        
