from preprocessing import *
from data.conect2db import *
from utils.Utils import options
from data.DataLoader import MiniBatchLoader

options_dic = readDictionary("../src/properties/options.properties")
fpath = readDictionary("../src/properties/fpath.properties")

print("loading data set from database ........")
test_dataset = getDatasetfromDB(
    vocabDic_path=fpath['vocabDic_path'],
    properties_path=fpath['properties_path']
)
print("loading successfully!")

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