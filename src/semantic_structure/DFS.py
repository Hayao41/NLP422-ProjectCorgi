import SemanticStructure as sStructure
import TreeModel as tm
import EmbeddingLayer as embed
import ContextEncoder as encoder
import Utils
from Utils import options
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

word = "I like this dog".split()

pos = ["VV",  "NN", "CON"]

relation = ["dsubj", "dobj", "conj"]

word2ix = Utils.make_dictionary(word)

pos2ix = Utils.make_dictionary(pos)

rel2ix = Utils.make_dictionary(relation)

print(word2ix)
print(pos2ix)
print(rel2ix)

# like
root = sStructure.SemanticGraphNode(word[1],
                                    pos[0],
                                    2,
                                    word_idx=word2ix[word[1]],
                                    label=1,
                                    pos_idx=pos2ix[pos[0]])
# I
node1 = sStructure.SemanticGraphNode(word[0],
                                     pos[1],
                                     1,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2ix[word[0]],
                                     pos_idx=pos2ix[pos[1]])
# this
node2 = sStructure.SemanticGraphNode(word[2],
                                     pos[2],
                                     3,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2ix[word[2]],
                                     pos_idx=pos2ix[pos[2]])
# dog
node3 = sStructure.SemanticGraphNode(word[3],
                                     pos[1],
                                     4,
                                     label=0,
                                     word_idx=word2ix[word[3]],
                                     pos_idx=pos2ix[pos[1]])

sentence = []
sentence.append(node1)
sentence.append(root)
sentence.append(node2)
sentence.append(node3)

edge1 = sStructure.SemanticGraphEdge(root,
                                     node1,
                                     relation[0],
                                     rel_idx=rel2ix[relation[0]])
edge2 = sStructure.SemanticGraphEdge(root,
                                     node3,
                                     relation[1],
                                     rel_idx=rel2ix[relation[1]])
edge3 = sStructure.SemanticGraphEdge(node3,
                                     node2,
                                     relation[2],
                                     rel_idx=rel2ix[relation[2]])

outedge_list1 = [edge1, edge2]
outedge_list2 = [edge3]

inedge_list1 = [edge1]
inedge_list2 = [edge2]
inedge_list3 = [edge3]

graph = sStructure.SemanticGraph(root)
outgoing_edges = {}
outgoing_edges[root] = outedge_list1
outgoing_edges[node3] = outedge_list2
graph.outgoing_edges = outgoing_edges

incoming_edges = {}
incoming_edges[node1] = inedge_list1
incoming_edges[node3] = inedge_list2
incoming_edges[node2] = inedge_list3
graph.incoming_edges = incoming_edges

graph.indexedWords = sentence

options = options(
    word_vocab_size=len(word2ix),
    word_emb_dims=20,
    pos_vocab_size=len(pos2ix),
    pos_emb_dims=20,
    rel_vocab_size=len(rel2ix),
    rel_emb_dims=0,
    context_linear_dim=20,
    bi_hid_dims=10,
    l_hid_dims=5,
    r_hid_dims=5,
    xavier=True
)

print(options.pos_vocab_size)

embed_model = embed.EmbeddingLayer(options=options)
encoder_model = encoder.ContextEncoder(options=options)
tree_model = tm.HierarchicalTreeLSTMs(options=options)
# tree_model = tm.DynamicRecursiveNetwork(options=options)

model = tm.TestModel(embed=embed_model, tree_model=tree_model, encoder=encoder_model)

print(model.parameters())

model(graph)

loss_function = nn.CrossEntropyLoss(size_average=True)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)

# adam is better(as the author says (玄学))
optimizer = optim.Adam(model.parameters(), lr=0.001)

e_list =[]
l_list = []

TRAIN1 = False
TRAIN2 = True

if TRAIN1:

    for epoch in range(100):

        # for parameter in model.parameters():
        #     print(parameter)

        e_list.append(epoch)

        target = Variable(torch.LongTensor(graph.getPOSIdxs()))

        model.zero_grad()

        tag_scores = model(graph)

        loss = loss_function(tag_scores, target)

        l_list.append(loss.data[0])

        loss.backward()

        optimizer.step()

    print(model(graph))

    plt.plot(e_list, l_list)
    plt.title('Test Context Encoder')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if TRAIN2:

    for epoch in range(1000):

        e_list.append(epoch)

        label = Variable(torch.LongTensor(graph.getLabel()))

        model.zero_grad()

        scores = model(graph)

        loss = loss_function(scores, label)

        l_list.append(loss.data[0])

        loss.backward()

        optimizer.step()

    print(model(graph))

    plt.plot(e_list, l_list)
    plt.title('Tree Encoder')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
