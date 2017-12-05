''' 
This file is semantic structure definition
 '''

class SemanticGraph(object):

    ''' 
    Semantic graph stores the dependency tree structure\n
    @Attribute :\n
    root : the root node of tree structure semantic graph\n
    outgoing_edges : outgoing edge map table '''

    def __init__(self, root=None):
        super(SemanticGraph, self).__init__()
        self.outgoing_edges = {}
        self.root = root


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
                 rel_vec=None):
        super(SemanticGraphEdge, self)
        self.source = source
        self.target = target
        self.relation = relation
        self.rel_vec = rel_vec


class SemanticGraphIterator(object):

    ''' 
    iterator for traversing semantic graph structue
    '''

    def __init__(self, node, graph):
        super(SemanticGraphIterator, self).__init__()
        self.node = node
        self.graph = graph
        self.c_list = self.getChildren()

    def getChildren(self):
        children_list = []
        if self.node in self.graph.outgoing_edges:
            edge_list=self.graph.outgoing_edges[self.node]
            for edge in edge_list:
                children_list.append(edge.target)
        return children_list

    def next(self):
        
        next_node = self.c_list[0]
        self.c_list.remove(next_node)
        return next_node

    def isLeaf(self):
        return self.node.isLeaf

    def allChildrenChecked(self):
        return len(self.c_list) == 0

