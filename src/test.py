
from utils.Utils import options
import torch.nn as nn
from models.Encoder import ContextEncoder
from models.TreeModel import HierarchicalTreeLSTMs
from models.SubLayer import MLP
from models.Detector import ClauseDetector
from models.SubLayer import TreeEmbedding
import torch.optim as optim
from matplotlib import pyplot as plt
from data.DataLoader import MiniBatchLoader
from preprocessing import *

options = options(
    word_vocab_size=len(word2idx),
    word_emb_dims=0,
    pos_vocab_size=len(pos2idx),
    pos_emb_dims=100,
    rel_vocab_size=len(rel2idx),
    rel_emb_dims=100,
    rp_vocab_size=20,
    rp_emb_dims=0,
    label_dims=len(label2idx),
    context_linear_dim=50,
    use_bi_lstm=True,
    lstm_hid_dims=50,
    lstm_num_layers=1,
    chain_hid_dims=50,
    batch_size=5,
    xavier=True,
    dropout=0.1,
    padding=0,
    use_cuda=False
)

test_dp1 = "-> like/VBP-2 (root)"\
        "\n  -> I/PRP-1 (nsubj:to)"\
        "\n  -> dog/NN-4 (dobj)"\
        "\n    -> this/DT-3 (det)"\
        "\n  -> ./.-5 (punct)"

test_dp2 = "-> loves/VBZ-2 (root)"\
        "\n   -> He/PRP-1 (nsubj)"\
        "\n   -> pen/NN-5 (dobj)"\
        "\n     -> that/IN-3 (det)"\
        "\n     -> colorful/JJ-4 (amod)"\
        "\n   -> ./.-6 (punct)"

test_dp3 = "-> went/VBD-3 (root)"\
        "\n  -> He/PRP-1 (nsubj)"\
        "\n  -> eventually/RB-2 (advmod)"\
        "\n  -> City/NNP-7 (nmod:to)"\
        "\n    -> to/TO-4 (case)"\
        "\n    -> New/NNP-5 (compound)"\
        "\n    -> York/NNP-6 (compound)"\
        "\n  -> and/CC-9 (cc)"\
        "\n  -> made/VBD-10 (conj:and)"\
        "\n    -> records/NNS-11 (dobj)"\
        "\n      -> Records/NNPS-14 (nmod:for)"\
        "\n        -> for/IN-12 (case)"\
        "\n        -> King/NNP-13 (compound)"\
        "\n    -> name/NN-17 (nmod:under)"\
        "\n      -> under/IN-15 (case)"\
        "\n      -> the/DT-16 (det)"\
        "\n      -> Grant/NNP-19 (dep)"\
        "\n        -> Al/NNP-18 (compound)"\
        "\n      -> one/CD-21 (dep)"\
        "\n        -> particular/JJ-23 (nmod:in)"\
        "\n          -> in/IN-22 (case)"\
        "\n          -> Cabaret/NN-26 (dep)"\
        "\n            -> appeared/VBN-29 (acl)"\
        "\n              -> charts/NNS-34 (nmod:in)"\
        "\n              -> in/IN-30 (case)"\
        "\n                -> the/DT-31 (det)"\
        "\n                -> Variety/NNP-32 (compound)"\
        "\n                -> magazine/NN-33 (compound)"

print(word2idx)
print(pos2idx)
print(rel2idx)

use_word = (options.word_emb_dims != 0)
use_pos = (options.pos_emb_dims != 0)
use_rel = (options.rel_emb_dims != 0)

test_graph1 = buildSemanticGraph(test_dp1, use_word=use_word, use_pos=use_pos, use_rel=use_rel, listLabel=[0], word2idx=word2idx, pos2idx=pos2idx, rel2idx=rel2idx)
test_graph2 = buildSemanticGraph(test_dp2, use_word=use_word, use_pos=use_pos, use_rel=use_rel, listLabel=[0], word2idx=word2idx, pos2idx=pos2idx, rel2idx=rel2idx)
test_graph3 = buildSemanticGraph(test_dp3, use_word=use_word, use_pos=use_pos, use_rel=use_rel, listLabel=[0], word2idx=word2idx, pos2idx=pos2idx, rel2idx=rel2idx)

data1 = DataTuple(indexedWords=test_graph1.indexedWords, graph=test_graph1)
data2 = DataTuple(indexedWords=test_graph2.indexedWords, graph=test_graph2)
data3 = DataTuple(indexedWords=test_graph3.indexedWords, graph=test_graph3)

train_data_list = [data1, data2, data3, data1, data2, data3, data1, data2, data3]
test_data_list = [data1, data2, data3]

train_data = MiniBatchLoader(
    Dataset=train_data_list,
    shuffle=True,
    batch_size=4,
    use_cuda=options.use_cuda,
    use_word=use_word,
    use_pos=use_pos,
    has_graph=True
)

test_data = MiniBatchLoader(
    Dataset=test_data_list,
    shuffle=True,
    batch_size=1,
    use_cuda=options.use_cuda,
    use_word=use_word,
    use_pos=use_pos,
    has_graph=True
)


#===================  Train testing   ===================#
# detector module 
context_encoder = ContextEncoder(options=options)
tree_model = HierarchicalTreeLSTMs(options=options)
tree_embed = TreeEmbedding(options=options)
mlp = MLP(options=options)

# clause detector
detector = ClauseDetector(
    options=options, 
    context_encoder=context_encoder, 
    tree_embed=tree_embed, 
    tree_encoder=tree_model, 
    classifier=mlp
)

crit = nn.NLLLoss(size_average=True)
# optimizer = optim.SGD(detector.parameters(), lr=0.1, momentum=0.5)
# optimizer = optim.Adagrad(detector.parameters(), lr=1e-2, lr_decay=0.1)
optimizer = optim.Adam(detector.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

if options.use_cuda:
    detector.switch2gpu()
    crit = crit.cuda()

e_list = []
l_list = []

steps = 0

for epoch in range(20):
    
    for batch_index, batch_data in enumerate(train_data):

        e_list.append(steps)

        steps += 1

        sequneces, batch_graph, target_data = batch_data

        out = detector((sequneces, batch_graph)).outputs
    
        loss = crit(out, target_data)

        l_list.append(loss.cpu().data[0])

        loss.backward()

        optimizer.step()

for batch_data in test_data:

    sequneces, batch_graph, target_data = batch_data
    
    test_out = detector((sequneces, batch_graph)).preds

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
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Test Training Loss')
plt.show()

