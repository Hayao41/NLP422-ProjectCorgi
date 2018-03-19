
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

options_dic = readDictionary("../src/properties/options.properties")
fpath = readDictionary("../src/properties/fpath.properties")

test_dataset = conect2db.getDatasetfromDB(
    vocabDic_path=fpath['vocabDic_path'],
    properties_path=fpath['properties_path']
)

vocabDics = loadVocabDic(["pos", "rel", "act"], fpath['vocabDic_path'])
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
    label_dims=len(label2idx),
    pos_vocab_size=len(pos2idx),
    rel_vocab_size=len(rel2idx),
    word_emb_dims=options_dic['word_emb_dims'],
    pos_emb_dims=options_dic['pos_emb_dims'],
    rel_emb_dims=options_dic['rel_emb_dims'],
    rp_vocab_size=options_dic['rp_vocab_size'],
    rp_emb_dims=options_dic['rp_emb_dims'],
    context_linear_dim=options_dic['context_linear_dim'],
    use_bi_lstm=options_dic['use_bi_lstm'],
    lstm_hid_dims=options_dic['lstm_hid_dims'],
    lstm_num_layers=options_dic['lstm_num_layers'],
    chain_hid_dims=options_dic['chain_hid_dims'],
    batch_size=options_dic['batch_size'],
    xavier=options_dic['xavier'],
    dropout=options_dic['dropout'],
    padding=options_dic['padding'],
    use_cuda=options_dic['use_cuda']
)

use_word = (options.word_emb_dims != 0)
use_pos = (options.pos_emb_dims != 0)
use_rel = (options.rel_emb_dims != 0)

train_data_list = []
test_data_list = []

for data_item in test_dataset[0:-1]:
    data_tuple = DataTuple(indexedWords=data_item.indexedWords, graph=data_item)
    train_data_list.append(data_tuple)

for data_item in test_dataset[:2]:
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

for epoch in range(50):
    
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

    listLabel = []

    for word in batch_graph[0].indexedWords:
        listLabel.append(idx2label[word.label])

    print(test_out)

    insts_label = []

    for inst in test_out:
        inst_label = []
        for word in inst.cpu():
            word = word.data.numpy().tolist()
            inst_label.append(idx2label[word.index(max(word))])
        insts_label.append(inst_label)

    print(insts_label)
    print(listLabel)
    print(batch_graph[0])

plt.plot(e_list, l_list)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Test Training Loss')
plt.show()

