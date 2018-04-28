
import os
import gc
import time
import torch
import random
import preprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim
import data.conect2db as conect2db
from utils.Utils import options
from matplotlib import pyplot as plt
from models.Module import MLP
from models.Module import TreeEmbedding
from data.DataLoader import MiniBatchLoader
import models.Encoder as Encoder
from models.Detector import ClauseDetector
from models.TreeModel import HierarchicalTreeLSTMs
from models.TreeModel import DynamicRecursiveNetwork

random.seed(time.time())
torch.manual_seed(1024)


def build_model(options):
    
    ''' build model, loss function, optimizer controlled by options '''
    
    # detector module
    if options.use_bi_lstm:
        context_encoder = Encoder.ContextEncoder(options)
    else:
        context_encoder = Encoder.EmbeddingEncoder(options)

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
            momentum=options.momentum,
            weight_decay=options.weight_decay
        )
    elif options.optim == "Adagrad":
        optimizer = optim.Adagrad(
            detector.parameters(),  
            lr=options.lr, 
            lr_decay=options.lr_decay,
            weight_decay = options.weight_decay
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
    
    """ compute model's mini batch mean loss, scaled by down sample factor"""

    counts = []
    losses = []
    sample_mode = options.sample_mode

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

        neg_count, pos_count = count
        
        # get target data list
        target = target_data[1][target_index]
    
        # get the negative sample loss scale factor(inner sentence)
        if options.sample_prop == -1:
            scale_factor = 1

        elif pos_count == 0:
            scale_factor = 1

        elif (neg_count / pos_count) <= options.sample_prop:
            scale_factor = 1

        else:
            scale_factor = (pos_count * options.sample_prop) / neg_count

        # get sentence losses
        inst_losses = crit(output[target_index], target)

        # scale down neg sample loss
        for inst_index, loss in enumerate(inst_losses):

            # scale sample loss by factor
            if "down" == sample_mode and target.data[inst_index] == 0:
                loss = loss * scale_factor
            elif "up" == sample_mode and target.data[inst_index] == 1:
                loss = loss / scale_factor

            losses.append(loss)

    # mini batch mean loss
    batch_mean_loss = sum(losses) / len(losses)

    return batch_mean_loss


def get_performance(outputs, targets):
    
    """ compute model's metrics """

    _, preds = torch.max(outputs, -1)
    
    TP = ((preds.cpu().data == 1) & (targets.cpu().data == 1)).cpu().sum()
    TN = ((preds.cpu().data == 0) & (targets.cpu().data == 0)).cpu().sum()
    FN = ((preds.cpu().data == 0) & (targets.cpu().data == 1)).cpu().sum()
    FP = ((preds.cpu().data == 1) & (targets.cpu().data == 0)).cpu().sum()

    return TP, TN, FN, FP


def epoch_train(training_batches, model, crit, optimizer, epoch, options):
    
    ''' train stage at one epoch '''

    step_losses = []
    
    # switch to train mode
    model.train()

    # initialize context_encoder's hidden states with train_batch_size
    if options.use_bi_lstm:
        model.context_encoder.init_hidden(options.train_batch_size)

    for batch_index, batch_data in enumerate(training_batches):

        start_batch = time.time()

        # prepare data(sampled from MiniBatchLoader)
        sequences, batch_graph, target_data = batch_data

        # clear paramter's grad
        model.zero_grad()
        optimizer.zero_grad()

        # repack lstm's hidden states
        if options.use_bi_lstm:
            model.context_encoder.repackage_hidden()

        # get batch output
        outputs, preds = model((sequences, batch_graph))

        # get model's cost in this stage
        loss = get_cost(preds, target_data, crit, options)

        # call back to compute gradient
        loss.backward()

        # update model's parameters
        optimizer.step()

        loss_data = loss.cpu().data[0]

        step_losses.append(loss_data)

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

    return step_losses


def epoch_test(test_batches, model, crit, options):
    
    """ eval stage at one epoch's end """
    
    # switch model to eval mode
    model.eval()

    # initialize context_encoder's hidden states with train_batch_size
    if options.use_bi_lstm:
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

        # get batch output
        outputs, preds = model((sequences, batch_graph))

        # get test batch loss
        loss = get_cost(preds, target_data, crit, options)
        test_loss += loss.cpu().data[0]

        # repack lstm's hidden states
        if options.use_bi_lstm:
            model.context_encoder.repackage_hidden()

        # get model's perfomance in this stage
        batch_TP, batch_TN, batch_FN, batch_FP = get_performance(outputs, target_data[0])

        TP += batch_TP
        FP += batch_FP
        TN += batch_TN
        FN += batch_FN

        # clean up trees
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

    test_mean_loss = test_loss / (batch_index + 1) 

    print("Epoch Test Metrics: Test ACC[{:.2f}%] Test Loss[{:.6f}] \n\t\t\t\t\tP[{:.2f}%], R[{:.2f}%], F1[{:.2f}%]\n".format(
        (acc * 100), test_mean_loss, (p * 100), (r * 100), (F1 * 100)
    ))

    return acc, p, r, F1, test_mean_loss


def saveModel(model, metrics, options):
    
    """ save model state into file for using or evaluating """

    acc, p, r, F1, valid_acc, epoch, local_time = metrics

    model_state = model.state_dict()
    tree_type = options.tree_type
    model_path = options.model_path + options.save_mode + "/"
    use_lstm = options.use_bi_lstm
    use_tree = options.use_tree
    tree_dir = options.direction
    opt_type = options.optim

    checkpoint = {
        'model': model_state,
        'settings': options,
        'epoch': epoch
    }

    if options.save_mode == "all":
        if use_tree:
            if use_lstm:
                model_path += "use_bi_lstm/"
            else:
                model_path += "no_bi_lstm/"

            if tree_type == "DRN":
                model_path += tree_type + "_" + tree_dir + "/" + opt_type + "/" + local_time
                model_name = model_path + '/accu_{accu:3.3f}_epoch_at_{epoch}.chkpt'.format(accu=100 * acc, epoch=epoch)
            else:
                model_path += tree_type + "/" + opt_type + "/" + local_time
                model_name = model_path + '/accu_{accu:3.3f}_epoch_at_{epoch}.chkpt'.format(accu=100 * acc, epoch=epoch)

        else:
            model_path += "pure_rnn" + "/" + opt_type + "/" + local_time
            model_name = model_path + '/accu_{accu:3.3f}_epoch_at_{epoch}.chkpt'.format(accu=100 * acc, epoch=epoch)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(checkpoint, model_name)

    elif options.save_mode == "best":
        if use_tree:
            if use_lstm:
                model_path += "use_bi_lstm/"
            else:
                model_path += "no_bi_lstm/"

            if tree_type == "DRN":
                model_path += tree_type + "_" + tree_dir + "/" + opt_type + "/" +  local_time
                model_name = model_path + '/best.chkpt'.format(accu=100 * acc, epoch=epoch)
            else:
                model_path += tree_type + "/" + opt_type + "/" + local_time
                model_name = model_path + '/best.chkpt'.format(accu=100 * acc, epoch=epoch)
        else:
            model_path += "pure_rnn" + "/" + opt_type + "/" + local_time
            model_name = model_path + '/best.chkpt'.format(accu=100 * acc, epoch=epoch)

        if acc >= max(valid_acc):
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            torch.save(checkpoint, model_name)
            print('    - [Info] The checkpoint file has been updated.')


def saveMetrics(metrics, step_losses, local_time, options):
    
    """ save model eval metrics into file """
    
    tree_type = options.tree_type
    tree_dir = options.direction
    log_path = options.log_path
    use_lstm = options.use_bi_lstm
    opt_type = options.optim

    if options.use_tree:
        if use_lstm:
            log_path += "use_bi_lstm/"
        else:
            log_path += "no_bi_lstm/"

        if tree_type == "DRN":
            log_path += tree_type + "_" + tree_dir + "/" + opt_type + "/" + local_time
        else:
            log_path += tree_type + "/" + opt_type + "/" + local_time
    else:
        log_path += "pure_rnn" + "/" + opt_type + "/" + local_time

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    metric_log_file = log_path + "/" + "metrics.log"
    loss_log_file = log_path + "/" + "step_loss.log"

    print("    - [Info] Saving metrics into file " + metric_log_file)
    print("    - [Info] Saving step losses into file " + loss_log_file)
    
    with open(metric_log_file, mode="w", encoding="utf-8") as me_log,\
            open(loss_log_file, mode="w", encoding="utf-8") as loss_log:

        loss_log.write("training step losses\n")
        [loss_log.write("{:.6f}".format(loss) + "\n") for loss in step_losses]

        me_log.write("epoch, test_loss, acc, p, r, F1\n")
        for metric in metrics:
            acc, p, r, F1, test_loss, epoch = metric
            log = "{epoch}, {loss:.6f}, {acc:3.3f}, {p:3.3f}, {r:3.3f}, {F1:3.3f}".format(
                epoch=epoch,
                loss=test_loss,
                acc=acc*100,
                p=p*100,
                r=r*100,
                F1=F1*100
            )
            me_log.write(log + "\n")


def saveTestID(test_set, test_id_path):
    
    """ save test set's pid into file for angeli's testing"""

    local_time = time.strftime("%Y-%m-%d", time.localtime())

    test_id_path += local_time + "/"

    if not os.path.exists(test_id_path):
        os.makedirs(test_id_path)

    test_id_path += "test_set_ID.txt"

    with open(test_id_path, "w", encoding="utf-8") as tid:
        tid.write("test set data base pid\n")
        [tid.write(inst.graph.sid + "\n") for inst in test_set]


def plotMetrics(metrics, step_losses, local_time, options):

    tree_type = options.tree_type
    tree_dir = options.direction
    pic_path = options.pic_path
    use_lstm = options.use_bi_lstm
    opt_type = options.optim

    if options.use_tree:
        if use_lstm:
            pic_path += "use_bi_lstm/"
        else:
            pic_path += "no_bi_lstm/"

        if tree_type == "DRN":
            pic_path += tree_type + "_" + tree_dir + "/" + opt_type + "/" + local_time + "/"
        else:
            pic_path += tree_type + "/" + opt_type + "/" + local_time + "/"
    else:
        pic_path += "pure_rnn" + "/" + opt_type + "/" + local_time + "/"

    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    metrics = np.array(metrics)
    acc = metrics[:, 0]
    p = metrics[:, 1]
    r = metrics[:, 2]
    F1 = metrics[:, 3]
    test_loss = metrics[:, 4]
    epoch = metrics[:, 5].astype(np.int32)
    steps = np.arange(0, len(step_losses), 1)

    plot_pic(
        title="Training Step Loss",
        x_content=steps,
        y_content=step_losses,
        xlabel="Step",
        ylabel="Loss",
        xlim=(0, steps[-1]),
        path=pic_path + "train_loss.svg"
    )

    plot_pic(
        title="Testing Epoch Loss",
        x_content=epoch,
        y_content=test_loss,
        xlabel="Epoch",
        ylabel="Loss",
        xlim=(0, epoch[-1]),
        path=pic_path + "test_loss.svg"
    )

    plot_pic(
        title="Testing Epoch Precision",
        x_content=epoch,
        y_content=p,
        xlabel="Epoch",
        ylabel="Precision",
        xlim=(0, epoch[-1]),
        path=pic_path + "test_p.svg"
    )

    plot_pic(
        title="Testing Epoch Recall",
        x_content=epoch,
        y_content=r,
        xlabel="Epoch",
        ylabel="Recall",
        xlim=(0, epoch[-1]),
        path=pic_path + "test_r.svg"
    )

    plot_pic(
        title="Testing Epoch F1",
        x_content=epoch,
        y_content=F1,
        xlabel="Epoch",
        ylabel="F1",
        xlim=(0, epoch[-1]),
        path=pic_path + "test_F1.svg"
    )

    plot_pic(
        title="Testing Accuracy",
        x_content=epoch,
        y_content=acc,
        xlabel="Epoch",
        ylabel="Accuracy",
        xlim=(0, epoch[-1]),
        path=pic_path + "test_acc.svg"
    )


def plot_pic(title, x_content, y_content, xlabel, ylabel, xlim, path):

    print("    - [Info] Plotting metrics into picture " + path)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = True

    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")
    plt.xlim(xlim)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.plot(x_content, y_content)
    plt.xlabel(xlabel, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=13, fontweight='bold')
    plt.savefig(path, format='svg')
    plt.clf()
    

def train(training_batches, test_batches, model, crit, optimizer, options):
    
    """ start training """

    local_time = time.strftime("%Y-%m-%d", time.localtime())
    step_losses = []
    valid_acc = []
    metrics = []

    if options.tree_type == "HTLstms":
        model.tree_encoder.init_hidden()

    for epoch in range(options.epoch):

        start_epoch = time.time()

        print("\nTraining on Epoch[{}]:".format(epoch))
        # epoch training
        step_losses += epoch_train(training_batches, model, crit, optimizer, epoch, options)

        end_epoch = time.time()

        time_epoch = end_epoch - start_epoch

        print("Epoch[{epoch}] time cost [{cost_s:.3f} Sec/{cost_m:.2f} Min/{cost_h:.2f} Hou]".format(
            epoch=epoch,
            cost_s=time_epoch,
            cost_m=(time_epoch / 60),
            cost_h=((time_epoch / 60) / 60)
            
        ))

        if epoch % 1 == 0 or epoch == (options.epoch - 1):

            # training stage eval
            acc, p, r, F1, test_loss = epoch_test(test_batches, model, crit, options)

            metrics.append([acc, p, r, F1, test_loss, epoch])

            valid_acc += [acc]

            if options.save_model:
                saveModel(model, (acc, p, r, F1, valid_acc, epoch, local_time), options)

    saveMetrics(metrics, step_losses, local_time, options)

    plotMetrics(metrics, step_losses, local_time, options)

    print("    - [Info] Training stage ends")


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
        use_non_linear_trans=options_dic['use_non_linear_trans'],
        context_linear_dim=options_dic['context_linear_dim'],
        inner_hidden_dims=options_dic['inner_hidden_dims'],
        

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
        direction=options_dic['direction'],

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
        sample_prop=options_dic['sample_prop'],
        sample_mode=options_dic['sample_mode'],
        save_model=options_dic['save_model'],
        save_mode=options_dic['save_mode'],
        model_path=fpath['model_path'],
        log_path=fpath['log_path'],
        pic_path=fpath['pic_path'],

        # data set prop
        train_prop=options_dic['train_prop'],
        test_prop=options_dic['test_prop'],
        dev_prop=options_dic['dev_prop']
    )

    # set whether use item
    use_word = (options.word_emb_dims != 0)
    use_pos = (options.pos_emb_dims != 0)
    use_rel = (options.rel_emb_dims != 0)
    use_rp = (options.rp_emb_dims != 0)

    # load annotated dataset from database then shuffle it
    if "full" == options_dic['data_load_mode']:
        if options_dic['test_mode']:
            annotated_dataset = conect2db.data_load_test(
                                        vocabDic_path=fpath['vocabDic_path'],
                                        properties_path=fpath['properties_path']
                                    )
        else:
            annotated_dataset = conect2db.getDatasetfromDB(
                                    vocabDic_path=fpath['vocabDic_path'],
                                    properties_path=fpath['properties_path']
                                )

        random.shuffle(annotated_dataset)

        if options_dic['test_prob']:
            training_set, test_set, _ = preprocessing.splitTestDataSet(
                train=options.train_prop,
                test=options.test_prop,
                develop=options.dev_prop,
                dataset=annotated_dataset
            )
        else:
            training_set, test_set, _ = preprocessing.splitDataSet(
                train=options.train_prop,
                test=options.test_prop,
                develop=options.dev_prop,
                dataset=annotated_dataset
            )

        saveTestID(test_set, options_dic['test_id_path'])
    else:
        training_set, test_set = conect2db.splited_load_dataset(
            vocabDic_path=fpath['vocabDic_path'],
            properties_path=fpath['properties_path'],
            test_id_path=fpath['test_id_path']
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
