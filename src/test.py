import semantic.SemanticStructure as sStructure
import utils.Utils as Utils
from utils.Utils import options
import models.TreeModel as TreeNN
import torch.nn as nn
import torch.nn.functional as F
from models.ContextEncoder import ContextEncoder
import torch.optim as optim
import torch
from torch.autograd import Variable
import utils.Constants as Constants
import numpy as np
from matplotlib import pyplot as plt

word = "I like this dog".split() + "He loves that colorful pen.".split()

pos = ["VV",  "NN", "IN", "JJ"]

relation = ["dsubj", "dobj", "det", "amod"]

word_pad = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK
}

pos_pad = {
    Constants.PAD_POS: Constants.PAD,
    Constants.UNK_POS: Constants.UNK
}

rel_pad = {
    Constants.UNK_REL: Constants.UNK
}

word2ix = Utils.make_dictionary(word, word_pad)

ix2word = {ix : word for word, ix in word2ix.items()}

pos2ix = Utils.make_dictionary(pos, pos_pad)

ix2pos = {ix : pos for pos, ix in pos2ix.items()}

rel2ix = Utils.make_dictionary(relation, rel_pad)

print(word2ix)
print(pos2ix)
print(rel2ix)

options = options(
    word_vocab_size=len(word2ix),
    word_emb_dims=20,
    pos_vocab_size=len(pos2ix),
    pos_emb_dims=0,
    rel_vocab_size=len(rel2ix),
    rel_emb_dims=0,
    context_linear_dim=20,
    use_bi_lstm=True,
    lstm_hid_dims=10,
    lstm_direction=2,
    lstm_num_layers=1,
    chain_hid_dims=20,
    batch_size=2,
    xavier=True,
    dropout=0.1,
    cuda=False
)


def make_training_data(indexwords_list):
    words_list = [[word.word_idx for word in words] for words in indexwords_list]
    pos_list = [[word.pos_idx for word in words] for words in indexwords_list]

    max_len = max(len(inst) for inst in words_list)

    word_data = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in words_list
    ])

    pos_data = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in pos_list
    ])

    return word_data, pos_data


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


class TestModel(nn.Module):
    def __init__(self, options, encoder):
        super(TestModel, self).__init__()

        self.encoder = encoder

        self.out_dims = options.pos_vocab_size

        self.hid2tag = nn.Linear(options.lstm_hid_dims, self.out_dims)

    def switch2gpu(self):
        self.encoder.switch2gpu()
        self.cuda()

    def forward(self, sequences):
        out = self.encoder(sequences)
        out = self.hid2tag(out)

        return F.log_softmax(out, dim=-1)


temp = Variable(torch.zeros(1, options.lstm_hid_dims))
if options.cuda:
    temp = temp.cuda()

# like
root1 = sStructure.SemanticGraphNode(word[1],
                                    pos[0],
                                    2,
                                    word_idx=word2ix[word[1]],
                                    label=1,
                                    context_vec=temp,
                                    pos_idx=pos2ix[pos[0]])
# I
node1 = sStructure.SemanticGraphNode(word[0],
                                     pos[1],
                                     1,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2ix[word[0]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[1]])
# this
node2 = sStructure.SemanticGraphNode(word[2],
                                     pos[2],
                                     3,
                                     label=0,
                                     isLeaf=True,
                                     word_idx=word2ix[word[2]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[2]])
# dog
node3 = sStructure.SemanticGraphNode(word[3],
                                     pos[1],
                                     4,
                                     label=0,
                                     word_idx=word2ix[word[3]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[1]])

# love
root2 = sStructure.SemanticGraphNode(word[5],
                                     pos[0],
                                     2,
                                     label=1,
                                     word_idx=word2ix[word[5]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[0]])

# he
node4 = sStructure.SemanticGraphNode(word[4],
                                     pos[1],
                                     1,
                                     label=0,
                                     word_idx=word2ix[word[4]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[1]])

# that
node5 = sStructure.SemanticGraphNode(word[6],
                                     pos[2],
                                     3,
                                     label=0,
                                     word_idx=word2ix[word[6]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[2]])

# colorful
node6 = sStructure.SemanticGraphNode(word[7],
                                     pos[3],
                                     4,
                                     label=0,
                                     word_idx=word2ix[word[7]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[3]])

# pen
node7 = sStructure.SemanticGraphNode(word[8],
                                     pos[1],
                                     5,
                                     label=0,
                                     word_idx=word2ix[word[8]],
                                     context_vec=temp,
                                     pos_idx=pos2ix[pos[1]])


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
                                     rel_idx=rel2ix[relation[0]])

# like -> dog
edge2 = sStructure.SemanticGraphEdge(root1,
                                     node3,
                                     relation[1],
                                     rel_idx=rel2ix[relation[1]])

# dog -> this
edge3 = sStructure.SemanticGraphEdge(node3,
                                     node2,
                                     relation[2],
                                     rel_idx=rel2ix[relation[2]])

# love -> he
edge4 = sStructure.SemanticGraphEdge(root2,
                                     node4,
                                     relation[0],
                                     rel_idx=rel2ix[relation[0]])

# love -> pen
edge5 = sStructure.SemanticGraphEdge(root2,
                                     node7,
                                     relation[1],
                                     rel_idx=rel2ix[relation[1]])

# pen -> that
edge6 = sStructure.SemanticGraphEdge(node7,
                                     node5,
                                     relation[2],
                                     rel_idx=rel2ix[relation[2]])

# pen -> colorful
edge7 = sStructure.SemanticGraphEdge(node7,
                                     node6,
                                     relation[3],
                                     rel_idx=rel2ix[relation[3]])


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

word_data, pos_data = make_training_data(indexwords_list)

batch_word_data = Variable(torch.from_numpy(word_data))
batch_pos_data = Variable(torch.from_numpy(pos_data))

sequences = Sequences(words=batch_word_data, pos=batch_pos_data, batch_size=len(indexwords_list))

print(batch_word_data, batch_pos_data)

encoder = ContextEncoder(options=options)
test = TestModel(options=options, encoder=encoder)
crit = nn.NLLLoss(size_average=True)
# optimizer = optim.SGD(test.parameters(), lr=0.1, momentum=0.3)
optimizer = optim.Adam(test.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

test_data = Variable(torch.from_numpy(word_data))

tree = TreeNN.HierarchicalTreeLSTMs(options=options)

if options.cuda:
    sequences.switch2gpu()
    test.switch2gpu()
    crit = crit.cuda()
    test_data = test_data.cuda()
    tree.switch2gpu()

print(test_data)
tree(graph1)
print(graph1)

RUN = True

if RUN:

    e_list = []
    l_list = []

    for epoch in range(1000):
        e_list.append(epoch)

        test.zero_grad()
        out = test(sequences)

        loss = 0


        loss = crit(out.view(-1, len(pos2ix)), sequences.pos.view(-1))

        l_list.append(loss.data[0])

        loss.backward()
        optimizer.step()

    test_seq = Sequences(words=test_data, batch_size=2)
    if options.cuda:
        test_seq.switch2gpu()

    test_out = test(test_seq)

    insts_pos = []
    for inst in test_out.cpu():
        inst_pos = []
        for word in inst:
            word = word.data.numpy().tolist()
            inst_pos.append(ix2pos[word.index(max(word))])
        insts_pos.append(inst_pos)

    print(insts_pos)

    plt.plot(e_list, l_list)
    plt.title('Test Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()