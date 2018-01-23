''' 
This file is semantic structure definition
 '''
import torch
from collections import namedtuple

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
                indexedWords=None
                ):
        super(SemanticGraph, self).__init__()
        self.root = root
        self.outgoing_edges = outgoing_edges
        self.incoming_edges = incoming_edges
        self.indexedWords = indexedWords

    def getLabels(self):
        
        for word in self.indexedWords:
            yield word.label

    def outcomingEdges(self):
        
        for source in self.outgoing_edges:
            edges = self.outgoing_edges[source]
            for edge in edges:
                yield edge

    def incomingEdges(self):
        
        for target in self.incoming_edges:
            edges = self.incoming_edges[target]
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
    
    def __len__(self):
        return len(self.indexedWords)
    

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
                label=0,
                isLeaf=False
    ):
        super(SemanticGraphNode, self).__init__()
        self.text = text
        self.pos = pos
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.word_idx = word_idx
        self.pos_idx = pos_idx
        self.rp_idx = rp_idx
        self.rp_vec = rp_vec
        self.context_vec = context_vec
        self.label = label
        self.atten_prob = atten_prob
        self.isLeaf = isLeaf


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


class SemanticGraphIterator(object):

    '''
    iterator for traversing semantic graph structure
    '''

    def __init__(self, node, graph):
        super(SemanticGraphIterator, self).__init__()
        self.node = node
        self.graph = graph

        # list for recording unchecked children
        self.c_list = list(self.children())

    def getOutgoingEdges(self):
        if self.node in self.graph.outgoing_edges:
            for edge in self.graph.outgoing_edges[self.node]:
                yield edge
        else:
            # if there are no outgoing edges starts with current node, stop iteration
            raise StopIteration

    def getIncomingEdges(self):
        if self.node in self.graph.incoming_edges:
            for edge in self.graph.incoming_edges[self.node]:
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
        
        IncomRelation = namedtuple('IncomRelation', ['relation', 'rel_idx', 'rel_vec'])
        
        for edge in self.getIncomingEdges():
            yield IncomRelation(
                relation=edge.relation,
                rel_idx=edge.rel_idx,
                rel_vec=edge.rel_vec
            )

    def left_hiddens(self):
        
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
