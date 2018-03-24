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
import preprocessing
import data.conect2db as conect2db
import time



if __name__ == "__main__":

    # load options from properties file
    options_dic = readDictionary("../src/properties/options.properties")
    fpath = readDictionary("../src/properties/fpath.properties")

    # load annotated dataset from database
    annotated_dataset = conect2db.getDatasetfromDB(
        vocabDic_path=fpath['vocabDic_path'],
        properties_path=fpath['properties_path']
    )

    # get vocabunary
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

    # build options
    options = options(

        # vocabunary size
        word_vocab_size=len(word2idx),
        label_dims=len(label2idx),
        pos_vocab_size=len(pos2idx),
        rel_vocab_size=len(rel2idx),

        # embedding layer params
        word_emb_dims=options_dic['word_emb_dims'],
        pos_emb_dims=options_dic['pos_emb_dims'],
        rel_emb_dims=options_dic['rel_emb_dims'],
        rp_emb_dims=options_dic['rp_emb_dims'],

        # non linear trans
        context_linear_dim=options_dic['context_linear_dim'],

        # context encoder
        use_bi_lstm=options_dic['use_bi_lstm'],
        lstm_num_layers=options_dic['lstm_num_layers'],
        lstm_hid_dims=options_dic['lstm_hid_dims'],
        
        # tree children chain
        use_bi_chain=options_dic['use_bi_chain'],
        chain_num_layers=options_dic['chain_num_layers'],
        chain_hid_dims=options_dic['chain_hid_dims'],

        # optimization
        batch_size=options_dic['batch_size'],
        xavier=options_dic['xavier'],
        dropout=options_dic['dropout'],
        padding=options_dic['padding'],
        use_cuda=options_dic['use_cuda']
    )

    # set if use item
    use_word = (options.word_emb_dims != 0)
    use_pos = (options.pos_emb_dims != 0)
    use_rel = (options.rel_emb_dims != 0)
    use_rp = (options.rp_emb_dims != 0)