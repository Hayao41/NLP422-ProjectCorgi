
import os
import gc
import time
import torch
import random
import preprocessing
import torch.nn as nn
import torch.optim as optim
import data.conect2db as conect2db
from matplotlib import pyplot as plt
from utils.Utils import options
from models.Module import MLP
from models.Module import TreeEmbedding
from data.DataLoader import MiniBatchLoader
from models.Encoder import ContextEncoder
from models.Detector import ClauseDetector
from models.TreeModel import HierarchicalTreeLSTMs
from models.TreeModel import DynamicRecursiveNetwork

random.seed(time.time())
torch.manual_seed(1024)


def build_model(options):
    
    ''' build model, loss function, optimizer determined by options '''
    
    # detector module
    context_encoder = ContextEncoder(options)

    if options.tree_type == "DRN":
        tree_model = DynamicRecursiveNetwork(options)
    else:
        tree_model = HierarchicalTreeLSTMs(options)

    tree_embed = TreeEmbedding(options)
    mlp = MLP(options)

    # clause detector
    detector = ClauseDetector(
        options=options, 
        context_encoder=context_encoder, 
        tree_embed=tree_embed, 
        tree_encoder=tree_model, 
        classifier=mlp
    )

    crit = nn.NLLLoss(reduce=options.loss_reduce)

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


def get_cost(output, target_data, crit, options):
    
    """ compute model's output and target's loss """
        
    counts = []
    losses = []

    # count neg and pos sample in one instance sentence
    for target in target_data[1]:
        neg_count = 0
        pos_count = 0
        for inst in target.data:
            if inst == 0:
                neg_count += 1
            else:
                pos_count += 1
        counts.append((neg_count, pos_count))

    for target_index, count in enumerate(counts):
        
        # get target data list
        target = target_data[1][target_index]
    
        # get the negative sample loss scale factor(inner sentence)
        if options.down_sample_prop == -1:
            scale_factor = 1

        elif count[1] == 0:
            scale_factor = 1

        elif (count[0] / count[1]) <= options.down_sample_prop:
            scale_factor = 1

        else:
            scale_factor = (count[1] * options.down_sample_prop) / count[0]

        # get sentence losses
        inst_losses = crit(output[target_index], target)

        # scale down neg sample loss
        for inst_index, loss in enumerate(inst_losses):

            # scale down negative sample loss by factor
            if target.data[inst_index] == 0:
                loss = loss * scale_factor

            losses.append(loss)

    # mini batch mean loss
    batch_mean_loss = sum(losses) / len(losses)

    return batch_mean_loss


def get_performance(outputs, targets):
    
    """ compute model metrics """

    _, preds = torch.max(outputs, -1)
    
    TP = ((preds.data == 1) & (targets.data == 1)).cpu().sum()
    TN = ((preds.data == 0) & (targets.data == 0)).cpu().sum()
    FN = ((preds.data == 0) & (targets.data == 1)).cpu().sum()
    FP = ((preds.data == 1) & (targets.data == 0)).cpu().sum()

    return TP, TN, FN, FP


def epoch_train(training_batches, model, crit, optimizer, epoch, options):
    
    ''' train stage in one epoch '''
    
    # switch to train mode
    model.train()

    # initialize context_encoder's hidden states with train_batch_size
    model.context_encoder.init_hidden(options.train_batch_size)

    for batch_index, batch_data in enumerate(training_batches):

        start_batch = time.time()

        # prepare data(sampled from MiniBatchLoader)
        sequences, batch_graph, target_data = batch_data

        # clear paramter's grad
        model.zero_grad()
        optimizer.zero_grad()

        # repack lstm's hidden states
        model.context_encoder.repackage_hidden()

        # get batch out put
        outputs, preds = model((sequences, batch_graph))

        # get model's cost in this stage
        loss = get_cost(preds, target_data, crit, options)

        # call back to compute gradient
        loss.backward()

        # update model's parameters
        optimizer.step()

        loss_data = loss.cpu().data[0]

        end_batch = time.time()

        [graph.clean_up() for graph in batch_graph]

        del batch_data, sequences, batch_graph
        del target_data, outputs, preds, loss

        gc.collect()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.5f} '.format(
                epoch, (batch_index + 1) * training_batches.batch_size, len(training_batches.dataset),
                (100. * ((batch_index + 1) / len(training_batches))), loss_data, end_batch - start_batch
            )
        )


def epoch_test(test_batches, model, crit, optimizer, epoch, options):
    
    """ eval stage at one epoch's end """
    
    # switch model to eval mode
    model.eval()

    # initialize context_encoder's hidden states with train_batch_size
    model.context_encoder.init_hidden(options.eval_batch_size)

    test_loss = 0

    # metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    p = 0
    r = 0
    F1 = 0
    acc = 0

    for batch_index, batch_data in enumerate(test_batches):
        
        # prepare data(sampled from MiniBatchLoader)
        sequences, batch_graph, target_data = batch_data

        # get batch out put
        outputs, preds = model((sequences, batch_graph))

        # get test batch loss
        loss = get_cost(preds, target_data, crit, options)
        test_loss += loss.cpu().data[0]

        # repack lstm's hidden states
        model.context_encoder.repackage_hidden()

        # get model's perfomance in this stage
        batch_TP, batch_TN, batch_FN, batch_FP = get_performance(outputs, target_data[0])

        TP += batch_TP
        FP += batch_FP
        TN += batch_TN
        FN += batch_FN

        [graph.clean_up() for graph in batch_graph]

        del batch_data, sequences, batch_graph
        del target_data, outputs, preds, loss

        gc.collect()

    # compute metrics
    if TP + FP != 0:
        p = TP / (TP + FP)

    if (TP + FN) != 0:
        r = TP / (TP + FN)
    
    if (r + p) != 0:
        F1 = 2 * r * p / (r + p)

    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Epoch Test Metrics: Test ACC[{:.2f}%] Test Loss[{:.6f}] \n\t\t\t\t\tP[{:.2f}%], R[{:.2f}%], F1[{:.2f}%]\n".format(
        (acc * 100), (test_loss / (batch_index + 1)), (p * 100), (r * 100), (F1 * 100)
    ))


def train(training_batches, test_batches, model, crit, optimizer, options):
    
    """ start training """

    if options.tree_type == "HTLstms":
        model.tree_encoder.init_hidden()

    for epoch in range(options.epoch):

        start_epoch = time.time()

        print("\nTraining on Epoch[{}]:".format(epoch))
        epoch_train(training_batches, model, crit, optimizer, epoch, options)

        epoch_test(test_batches, model, crit, optimizer, epoch, options)

        end_epoch = time.time()


def test_method(idx2label, model, test_batches, options):
    
    model.eval()
    model.context_encoder.init_hidden(options.eval_batch_size)
    if options.tree_type != "DRN":
        model.tree_encoder.init_hidden()
        
    for batch_data in test_batches:

        sequences, batch_graph, target_data = batch_data
        
        output = model((sequences, batch_graph))

        model.context_encoder.repackage_hidden()

        test_out = output.preds

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


if __name__ == "__main__":

    # load options from properties file
    options_dic = preprocessing.readDictionary("../src/properties/options.properties")
    fpath = preprocessing.readDictionary("../src/properties/fpath.properties")

    # set cuda devices
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

        # tree encoder type
        # @Type DRN : Dynamic recursive neural nets
        # @Type HTLstms : Hierarchical Tree LSTMs
        use_tree=options_dic['use_tree'],
        tree_type=options_dic['tree_type'],

        # attention
        atten_type=options_dic['atten_type'],

        # optimization
        train_batch_size=options_dic['train_batch_size'],
        eval_batch_size=options_dic['eval_batch_size'],
        epoch=options_dic['epoch'],
        xavier=options_dic['xavier'],
        dropout=options_dic['dropout'],
        padding=options_dic['padding'],
        use_cuda=options_dic['use_cuda'],
        cuda_device=options_dic['cuda_device'],
        optim=options_dic['optim'],
        lr=options_dic['lr'],
        lr_decay=options_dic['lr_decay'],
        weight_decay=options_dic['weight_decay'],
        momentum=options_dic['momentum'],
        betas=options_dic['betas'],
        eps=options_dic['eps'],
        loss_reduce=options_dic['loss_reduce'],
        down_sample_prop=options_dic['down_sample_prop'],

        # data set prop
        train_prop = options_dic['train_prop'],
        test_prop = options_dic['test_prop'],
        dev_prop = options_dic['dev_prop']
    )

    # set whether use item
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
    #                                 train=options.train_prop,
    #                                 test=options.test_prop,
    #                                 develop=options.dev_prop,
    #                                 dataset=annotated_dataset
    #                             )

    training_set, test_set, _ = preprocessing.splitTestDataSet(
        train=options.train_prop,
        test=options.test_prop,
        develop=options.dev_prop,
        dataset=annotated_dataset
    )

    # prepare mini batch loader
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
    model, crit, optimizer = build_model(options)

    # start training stage
    train(training_batches, test_batches, model, crit, optimizer, options)

    test_method(idx2label, model, test_batches, options)

