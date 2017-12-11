import SemanticStructure as sStructure
import HierarchicalTreeLSTMs as htlstm
from utils import options

word = "I like this dog".split()

# like
root = sStructure.SemanticGraphNode(word[1], "VV", 2)
# I
node1 = sStructure.SemanticGraphNode(word[0], "NN", 1, isLeaf=True)
# this
node2 = sStructure.SemanticGraphNode(word[2], "CON", 3, isLeaf=True)
# dog
node3 = sStructure.SemanticGraphNode(word[3], "NN", 4)

sentence = []
sentence.append(node1)
sentence.append(root)
sentence.append(node2)
sentence.append(node3)

edge1 = sStructure.SemanticGraphEdge(root, node1, "dsubj")
edge2 = sStructure.SemanticGraphEdge(root, node3, "dobj")
edge3 = sStructure.SemanticGraphEdge(node3, node2, "conj")

outedge_list1 = [edge1, edge2]
outedge_list2 = [edge3]

inedge_list1 = [edge1]
inedge_list2 = [edge2]
inedge_list3 = [edge3]

graph = sStructure.SemanticGraph(root)
graph.outgoing_edges[root] = outedge_list1
graph.outgoing_edges[node3] = outedge_list2

graph.incoming_edges[node1] = inedge_list1
graph.incoming_edges[node3] = inedge_list2
graph.incoming_edges[node2] = inedge_list3

graph.indexedWords = sentence

option = options(
    pos_vocab_size = 4,
    pos_emb_dims = 1,
    rel_vocab_size = 10,
    rel_emb_dims = 10,
    word_vocab_size = 10,
    word_emb_dims = 10,
    vec_dims = 10,
    bi_hid_dims = 10,
    l_hid_dims = 10,
    r_hid_dims = 10,
    rel_labeled_tag = 10
)

print(option.pos_vocab_size)

graph.indexedWords[0] = "do some changes"
model = htlstm.HierarchicalTreeLSTMs()
a = model(graph)
