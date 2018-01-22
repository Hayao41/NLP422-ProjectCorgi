import semantic.SemanticStructure as sStructure
import utils.Utils as Utils
from utils.Utils import options
import torch.nn as nn
from models.Encoder import ContextEncoder
from models.TreeModel import HierarchicalTreeLSTMs as hlstm
from models.SubLayer import MLP
from models.Detector import ClauseDetector
import torch.optim as optim
import torch
from torch.autograd import Variable
import utils.Constants as Constants
from matplotlib import pyplot as plt

word = "I like this dog".split() + "He loves that colorful pen.".split()

pos = ["VV",  "NN", "IN", "JJ"]

relation = ["dsubj", "dobj", "det", "amod"]

label = ["RECURSE", "SPLIT"]

word_pad = {
    Constants.PAD_WORD: Constants.PAD,
    Constants.UNK_WORD: Constants.UNK
}

pos_pad = {
    Constants.PAD_POS: Constants.PAD,
    Constants.UNK_POS: Constants.UNK
}

rel_pad = {
    Constants.UNK_REL: Constants.PAD
}

word2idx = Utils.make_dictionary(word, word_pad)

idx2word = {idx: word for word, idx in word2idx.items()}

pos2idx = Utils.make_dictionary(pos, pos_pad)

idx2pos = {idx: pos for pos, idx in pos2idx.items()}

rel2idx = Utils.make_dictionary(relation, rel_pad)

label2idx = Utils.make_dictionary(label, {})

idx2label = {idx: label for label, idx in label2idx.items()}

print(word2idx)
print(pos2idx)
print(rel2idx)

options = options(
    word_vocab_size=len(word2idx),
    word_emb_dims=50,
    pos_vocab_size=len(pos2idx),
    pos_emb_dims=0,
    rel_vocab_size=len(rel2idx),
    rel_emb_dims=0,
    label_dims=len(label2idx),
    context_linear_dim=30,
    use_bi_lstm=True,
    lstm_hid_dims=30,
    lstm_direction=2,
    lstm_num_layers=1,
    chain_hid_dims=30,
    batch_size=2,
    xavier=True,
    dropout=0.1,
    use_cuda=False
)


class Sequences(object):
    def __init__(self, words=None, pos=None, batch_size=1):

        self.words = words
        self.pos = pos
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def switch2gpu(self):
        if self.words is not None:
            self.words = self.words.cuda()
        if self.pos is not None:
            self.pos = self.pos.cuda()

    def switch2cpu(self):
        if self.words is not None:
            self.words = self.words.cpu()
        if self.pos is not None:
            self.pos = self.pos.cpu()


temp = Variable(torch.zeros(1, options.lstm_hid_dims))
if options.use_cuda:
    temp = temp.cuda()

# like
root1 = sStructure.SemanticGraphNode(word[1],
                                    pos[0],
                                    2,
                                    word_idx=word2idx[word[1]],
                                    label=1,
                                    context_vec=temp,
                                    pos_idx=pos2idx[pos[0]])
# I
node1 = sStructure.SemanticGraphNode(word[0],
                                     pos[1],
                                     1,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2idx[word[0]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[1]])
# this
node2 = sStructure.SemanticGraphNode(word[2],
                                     pos[2],
                                     3,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2idx[word[2]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[2]])
# dog
node3 = sStructure.SemanticGraphNode(word[3],
                                     pos[1],
                                     4,
                                     label=0,
                                     word_idx=word2idx[word[3]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[1]])

# love
root2 = sStructure.SemanticGraphNode(word[5],
                                     pos[0],
                                     2,
                                     label=1,
                                     word_idx=word2idx[word[5]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[0]])

# he
node4 = sStructure.SemanticGraphNode(word[4],
                                     pos[1],
                                     1,
                                     label=0,
                                     word_idx=word2idx[word[4]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[1]])

# that
node5 = sStructure.SemanticGraphNode(word[6],
                                     pos[2],
                                     3,
                                     label=0,
                                     word_idx=word2idx[word[6]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[2]])

# colorful
node6 = sStructure.SemanticGraphNode(word[7],
                                     pos[3],
                                     4,
                                     label=0,
                                     word_idx=word2idx[word[7]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[3]])

# pen
node7 = sStructure.SemanticGraphNode(word[8],
                                     pos[1],
                                     5,
                                     label=0,
                                     word_idx=word2idx[word[8]],
                                     context_vec=temp,
                                     pos_idx=pos2idx[pos[1]])


sentence1 = []
sentence1.append(node1)
sentence1.append(root1)
sentence1.append(node2)
sentence1.append(node3)

sentence2 = []
sentence2.append(node4)
sentence2.append(root2)
sentence2.append(node5)
sentence2.append(node6)
sentence2.append(node7)

# like -> I
edge1 = sStructure.SemanticGraphEdge(root1,
                                     node1,
                                     relation[0],
                                     rel_idx=rel2idx[relation[0]])

# like -> dog
edge2 = sStructure.SemanticGraphEdge(root1,
                                     node3,
                                     relation[1],
                                     rel_idx=rel2idx[relation[1]])

# dog -> this
edge3 = sStructure.SemanticGraphEdge(node3,
                                     node2,
                                     relation[2],
                                     rel_idx=rel2idx[relation[2]])

# love -> he
edge4 = sStructure.SemanticGraphEdge(root2,
                                     node4,
                                     relation[0],
                                     rel_idx=rel2idx[relation[0]])

# love -> pen
edge5 = sStructure.SemanticGraphEdge(root2,
                                     node7,
                                     relation[1],
                                     rel_idx=rel2idx[relation[1]])

# pen -> that
edge6 = sStructure.SemanticGraphEdge(node7,
                                     node5,
                                     relation[2],
                                     rel_idx=rel2idx[relation[2]])

# pen -> colorful
edge7 = sStructure.SemanticGraphEdge(node7,
                                     node6,
                                     relation[3],
                                     rel_idx=rel2idx[relation[3]])


outedge_list1 = [edge1, edge2]
outedge_list2 = [edge3]

outedge_list3 = [edge4, edge5]
outedge_list4 = [edge6, edge7]

inedge_list1 = [edge1]
inedge_list2 = [edge2]
inedge_list3 = [edge3]

inedge_list4 = [edge4]
inedge_list5 = [edge5]
inedge_list6 = [edge6]
inedge_list7 = [edge7]

graph1 = sStructure.SemanticGraph(root1)
outgoing_edges = {}
outgoing_edges[root1] = outedge_list1
outgoing_edges[node3] = outedge_list2
graph1.outgoing_edges = outgoing_edges

incoming_edges = {}
incoming_edges[node1] = inedge_list1
incoming_edges[node3] = inedge_list2
incoming_edges[node2] = inedge_list3
graph1.incoming_edges = incoming_edges

graph2 = sStructure.SemanticGraph(root2)
outgoing_edges2 = {}
outgoing_edges2[root2] = outedge_list3
outgoing_edges2[node7] = outedge_list4
graph2.outgoing_edges = outgoing_edges2

incoming_edges2 = {}
incoming_edges2[node4] = inedge_list4
incoming_edges2[node5] = inedge_list5
incoming_edges2[node6] = inedge_list6
incoming_edges2[node7] = inedge_list7
graph2.incoming_edges = incoming_edges2

graph1.indexedWords = sentence1
graph2.indexedWords = sentence2

print(list(graph1.getLabels()))
print(list(graph2.getLabels()))



indexwords_list = []
indexwords_list.append(sentence1)
indexwords_list.append(sentence2)

batch_word_data, batch_pos_data = Utils.make_training_data(indexwords_list)
target_data = Utils.make_target_data(indexwords_list)

sequences = Sequences(words=batch_word_data, pos=batch_pos_data, batch_size=len(indexwords_list))
batch_graph = [graph1, graph2]
batch_data = (sequences, batch_graph)

test_sequence = Sequences(words=batch_word_data, pos=batch_pos_data, batch_size=len(indexwords_list))
test_graph = [graph1, graph2]

test_data = (test_sequence, test_graph)


encoder = ContextEncoder(options=options)
tree_model = hlstm(options=options)
mlp = MLP(options=options)
test = ClauseDetector(options=options, encoder=encoder, tree=tree_model, classifier=mlp)

crit = nn.NLLLoss(size_average=True)
# optimizer = optim.SGD(test.parameters(), lr=0.1, momentum=0.6)
optimizer = optim.Adam(test.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)


if options.use_cuda:
    batch_data[0].switch2gpu()
    test_data[0].switch2gpu()
    test.switch2gpu()
    crit = crit.cuda()
    target_data = target_data.cuda()

RUN = True

if RUN:

    e_list = []
    l_list = []

    for epoch in range(100):
        e_list.append(epoch)

        test.zero_grad()
        out = test(batch_data).outputs

        loss = 0
        
        loss = crit(out, target_data)

        l_list.append(loss.cpu().data[0])

        loss.backward()
        optimizer.step()

    test_out = test(test_data).preds

    print(test_out)

    insts_label = []
    for inst in test_out:
        inst_label = []
        for word in inst.cpu():
            word = word.data.numpy().tolist()
            inst_label.append(idx2label[word.index(max(word))])
        insts_label.append(inst_label)

    print(insts_label)

    plt.plot(e_list, l_list)
    plt.title('Test Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

