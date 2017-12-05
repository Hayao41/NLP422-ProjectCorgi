import SemanticStructure as sStructure
import HierarchicalTreeLSTMs as htlstm

word = "I like this dog".split()

# like
root = sStructure.SemanticGraphNode(word[1], "VV", 2)
# I
node1 = sStructure.SemanticGraphNode(word[0], "NN", 1, isLeaf=True)
# this
node2 = sStructure.SemanticGraphNode(word[2], "CON", 3, isLeaf=True)
# dog
node3 = sStructure.SemanticGraphNode(word[3], "NN", 4)

edge1 = sStructure.SemanticGraphEdge(root, node1, "dsubj")
edge2 = sStructure.SemanticGraphEdge(root, node3, "dobj")
edge3 = sStructure.SemanticGraphEdge(node3, node2, "conj")

edge_list1 = [edge1, edge2]
edge_list2 = [edge3]

graph = sStructure.SemanticGraph(root)
graph.outgoing_edges[root] = edge_list1
graph.outgoing_edges[node3] = edge_list2

# stack = []
# root_ite = sStructure.SemanticGraphIterator(root, graph)
# stack.append(root_ite)
#
# while len(stack) is not 0:
#     ite = stack[len(stack)-1]
#     next = ite.next()
#     if next is not None:
#         next_ite = sStructure.SemanticGraphIterator(next, graph)
#         stack.append(next_ite)
#     else:
#         print(ite.node.text)
#         test = ite.getChildren()
#         stack.pop()

model = htlstm.HierarchicalTreeLSTMs()
model(graph)