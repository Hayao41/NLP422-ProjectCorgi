
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
import data.conect2db as conect2db

test_dataset = conect2db.getDatasetfromDB()

vocabDics = loadVocabDic(["pos", "rel", "act"], "/Users/joelchen/PycharmProjects/NLP422-ProjectCorgi/src/vocabDic/")
word2idx = vocabDics["word"]
pos2idx = vocabDics["pos"]
rel2idx = vocabDics["rel"]
label2idx = vocabDics["act"]

idx2word = {idx: inst for inst, idx in word2idx.items()}
idx2pos = {idx: inst for inst, idx in pos2idx.items()}
idx2rel = {idx: inst for inst, idx in rel2idx.items()}
idx2label = {idx: inst for inst, idx in label2idx.items()}

print("word dictionary: \n", word2idx)
print("pos dictionary: \n", pos2idx)
print("arc relation dictionary: \n", rel2idx)
print("action space dictionary: \n", label2idx)

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

use_word = (options.word_emb_dims != 0)
use_pos = (options.pos_emb_dims != 0)
use_rel = (options.rel_emb_dims != 0)

train_data_list = []
test_data_list = []

for data_item in test_dataset:
    data_tuple = DataTuple(indexedWords=data_item.indexedWords, graph=data_item)
    train_data_list.append(data_tuple)

for data_item in test_dataset[:3]:
    data_tuple = DataTuple(indexedWords=data_item.indexedWords, graph=data_item)
    test_data_list.append(data_tuple)

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

for epoch in range(30):
    
    for batch_index, batch_data in enumerate(train_data):

        e_list.append(steps)

        steps += 1

        sequences, batch_graph, target_data = batch_data

        out = detector((sequences, batch_graph)).outputs
    
        loss = crit(out, target_data)

        l_list.append(loss.cpu().data[0])

        loss.backward()

        optimizer.step()

for batch_data in test_data:

    sequences, batch_graph, target_data = batch_data
    
    test_out = detector((sequences, batch_graph)).preds

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

