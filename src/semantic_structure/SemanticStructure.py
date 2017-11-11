

#
# Semantic graph stores the dependency tree structure
# @Attribute :
#   root : the root node of tree structure semantic graph
#   outgoing_edges : outgoing edge map table
class SemanticGraph(object):
    def __init__(self, root=None):
        super(SemanticGraph, self).__init__()
        self.outgoing_edges = {}
        self.root = root


class SemanticGraphNode(object):
    def __init__(self,
                 text="",
                 pos="",
                 sentence_index=-1,
                 parent_index=-1,
                 word_vec=None,
                 pos_vec=None,
                 isLeaf=False,
                 isChecked=False):
        super(SemanticGraphNode, self)
        self.text = text
        self.pos = pos
        self.sentence_index = sentence_index
        self.parent_index = parent_index
        self.word_vec = word_vec
        self.pos_vec = pos_vec
        self.isLeaf = isLeaf
        self.isChecked = isChecked


class SemanticGraphEdge(object):
    def __init__(self,
                 source=None,
                 target=None,
                 relation="unknown",
                 rel_vec=None):
        super(SemanticGraphEdge, self)
        self.source = source
        self.target = target
        self.relation = relation
        self.rel_vec = rel_vec


word = "I like this dog".split()
root = SemanticGraphNode(word[1], "VV", 2)
node1 = SemanticGraphNode(word[0], "NN", 1, isLeaf=True)
node2 = SemanticGraphNode(word[2], "NN", 3)
node3 = SemanticGraphNode(word[3], "CON", 4, isLeaf=True)

edge1 = SemanticGraphEdge(root, node1, "dsubj")
edge2 = SemanticGraphEdge(root, node2, "dobj")
edge3 = SemanticGraphEdge(node2, node3, "conj")

edge_list1 = [edge1, edge2]
edge_list2 = [edge3]

graph = SemanticGraph(root)
graph.outgoing_edges[root] = edge_list1
graph.outgoing_edges[node2] = edge_list2

print(graph.outgoing_edges)

stack = []
stack.append(root)
stack.pop()







