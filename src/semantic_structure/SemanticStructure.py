''' 
This file is semantic structure definition
 '''


class SemanticGraph(object):

    ''' 
    Semantic graph stores the dependency tree structure\n
    @Attribute :\n
    root : the root node of tree structure semantic graph\n
    outgoing_edges : outgoing edge map table
    '''

    def __init__(self, 
                root=None,
                outgoing_edges=None,
                incoming_edges=None,
                indexedWords=None
                ):
        super(SemanticGraph, self).__init__()
        self.root = root
        self.outgoing_edges = {}
        self.incoming_edges = {}
        self.indexedWords = indexedWords
        


class SemanticGraphNode(object):
    
    def __init__(self,
                text=None,
                pos=None,
                sentence_index=None,
                parent_index=None,
                word_vec=None,
                pos_vec=None,
                hidden_vec=None,
                isLeaf=False,
                isChecked=False
                ):
        super(SemanticGraphNode, self)
        self.text = text
        self.pos = pos
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.word_vec = word_vec
        self.pos_vec = pos_vec
        self.hidden_vec = hidden_vec
        self.isLeaf = isLeaf
        self.isChecked = isChecked


class SemanticGraphEdge(object):
    
    def __init__(self,
                source=None,
                target=None,
                relation=None,
                rel_vec=None
                ):
        super(SemanticGraphEdge, self)
        self.source = source
        self.target = target
        self.relation = relation
        self.rel_vec = rel_vec


class SemanticGraphIterator(object):

    ''' iterator for traversing semantic graph structue '''

    def __init__(self, node, graph):
        super(SemanticGraphIterator, self).__init__()
        self.node = node
        self.graph = graph
        self.c_list = self.getChildren()

    def getChildren(self):
        children_list = []
        outgoingedge_list = self.getOutgoingEdges()
        for edge in outgoingedge_list:
            children_list.append(edge.target)
        return children_list

    def getParent(self):
        parent_list = []
        incomingedge_list = self.getIncomingEdges()
        for edge in incomingedge_list:
            parent_list.append(edge.source)
        return parent_list


    def getOutgoingEdges(self):
        if self.node in self.graph.outgoing_edges:
            return self.graph.outgoing_edges[self.node]
        else:
            # if there are no outgoing edges start with node return a empty list
            temp = []
            return temp

    def getIncomingEdges(self):
        if self.node in self.graph.incoming_edges:
            return self.graph.incoming_edges[self.node]
        else:
            temp = []
            return temp

    def next(self):
        next_node = self.c_list[0]
        next_ite = SemanticGraphIterator(next_node, self.graph)
        self.c_list.remove(next_node)
        return next_ite

    def isLeaf(self):
        if self.node.isLeaf is not None:
            return self.node.isLeaf
        else:
            return len(self.getChildren()) == 0

    def allChildrenChecked(self):
        # if children record list is empty, current node's children all have checked
        return len(self.c_list) == 0
        


