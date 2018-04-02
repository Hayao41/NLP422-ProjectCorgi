
import os
import gc
import time
import torch
import random
import objgraph
import linecache
import tracemalloc
import preprocessing
import torch.nn as nn
import torch.optim as optim
import data.conect2db as conect2db
from matplotlib import pyplot as plt
from utils.Utils import options
from models.SubLayer import MLP
from models.Encoder import ContextEncoder
from models.TreeModel import HierarchicalTreeLSTMs
from models.TreeModel import DynamicRecursiveNetwork
from models.Detector import ClauseDetector
from models.SubLayer import TreeEmbedding
from data.DataLoader import MiniBatchLoader

random.seed(time.time())
torch.manual_seed(1024)


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def buildModel(options):
    
    # detector module
    context_encoder = ContextEncoder(options)
    tree_model = DynamicRecursiveNetwork(options)
    tree_embed = TreeEmbedding(options)
    mlp = MLP(options)

    context_encoder.init_hidden(options.train_batch_size)

    # clause detector
    detector = ClauseDetector(
        options=options, 
        context_encoder=context_encoder, 
        tree_embed=tree_embed, 
        tree_encoder=tree_model, 
        classifier=mlp
    )

    crit = nn.NLLLoss(size_average=True)

    # get optimizer
    if options.optim == "SGD":
        optimizer = optim.SGD(
            detector.parameters(), 
            lr=options.lr, 
            momentum=options.momentum
        )
    elif options.optim == "Adagrad":
        optimizer = optim.Adagrad(
            detector.parameters(), 
            lr=options.lr, 
            lr_decay=options.lr_decay
        )
    else:
        optimizer = optim.Adam(
            detector.parameters(), 
            lr=options.lr, 
            betas=options.betas, 
            eps=options.eps, 
            weight_decay=options.weight_decay
        )

    if options.use_cuda:
        detector.switch2gpu()
        crit = crit.cuda()

    return detector, crit, optimizer


def epoch_train(training_batches, model, crit, optimizer, epoch):

    total_loss = 0.

    for batch_index, batch_data in enumerate(training_batches):

        start_batch = time.time()

        sequences, batch_graph, target_data = batch_data

        model.zero_grad()
        optimizer.zero_grad()

        model.context_encoder.repackage_hidden()

        out, _ = model((sequences, batch_graph))

        loss = crit(out, target_data)

        loss.backward()

        loss_data = loss.cpu().data[0]

        total_loss += loss_data

        optimizer.step()

        end_batch = time.time()

        for graph in batch_graph:
            graph.clean_up()

        del batch_data, sequences, batch_graph, target_data, out, loss

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.5f} '.format(
                epoch, (batch_index + 1) * training_batches.batch_size, len(training_batches.dataset),
                (100. * ((batch_index + 1) / len(training_batches))), loss_data, end_batch - start_batch
            )
        )

    return total_loss / (batch_index + 1)


def train(training_batches, test_batches, model, crit, optimizer, epoches):
    
    for epoch in range(epoches):
        
        start_epoch = time.time()

        print("\nTraining on Epoch[{}]:".format(epoch))
        mean_loss = epoch_train(training_batches, model, crit, optimizer, epoch)

        # epoch_test()

        end_epoch = time.time()

        # print("Ending on Epoch[{}] Batch mean loss[{:.6f}]\n".format(epoch, mean_loss))



if __name__ == "__main__":

    # load options from properties file
    options_dic = preprocessing.readDictionary("../src/properties/options.properties")
    fpath = preprocessing.readDictionary("../src/properties/fpath.properties")

    if options_dic['use_cuda']:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = options_dic['cuda_device']


    # get vocabulary
    vocabDics = preprocessing.loadVocabDic(["pos", "rel", "act"], fpath['vocabDic_path'])
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

        # attention
        atten_type=options_dic['atten_type'],

        # optimization
        train_batch_size=options_dic['train_batch_size'],
        epoch=options_dic['epoch'],
        xavier=options_dic['xavier'],
        dropout=options_dic['dropout'],
        padding=options_dic['padding'],
        use_cuda=options_dic['use_cuda'],
        cuda_device=options_dic['use_cuda'],
        optim=options_dic['optim'],
        lr=options_dic['lr'],
        lr_decay=options_dic['lr_decay'],
        weight_decay=options_dic['weight_decay'],
        momentum=options_dic['momentum'],
        betas=options_dic['betas'],
        eps=options_dic['eps']
    )

    # set if use item
    use_word = (options.word_emb_dims != 0)
    use_pos = (options.pos_emb_dims != 0)
    use_rel = (options.rel_emb_dims != 0)
    use_rp = (options.rp_emb_dims != 0)

    # load annotated dataset from database then shuffle it
    annotated_dataset = conect2db.getDatasetfromDB(
                            vocabDic_path=fpath['vocabDic_path'],
                            properties_path=fpath['properties_path']
                        )
    random.shuffle(annotated_dataset)

    # training_set, test_set, _ = preprocessing.splitDataSet(
    #                                 train=0.6, test=0.4, develop=0.0,
    #                                 dataset=annotated_dataset
    #                             )

    training_set, test_set, _ = preprocessing.splitDataSet(
                                    train=1, test=0.3, develop=0.0,
                                    dataset=annotated_dataset
                                )

    # prepare mini batches
    training_batches = MiniBatchLoader(
        Dataset=training_set,
        shuffle=True,
        batch_size=options.train_batch_size,
        use_cuda=options.use_cuda,
        use_word=use_word,
        use_pos=use_pos,
        has_graph=True
    )

    test_batches = MiniBatchLoader(
        Dataset=test_set,
        shuffle=True,
        batch_size=options.eval_batch_size,
        use_cuda=options.use_cuda,
        use_word=use_word,
        use_pos=use_pos,
        has_graph=True
    )

    # build model, loss_func and optim
    model, crit, optimizer = buildModel(options)



    train(
        training_batches,
        test_batches,
        model,
        crit,
        optimizer,
        options.epoch
    )


