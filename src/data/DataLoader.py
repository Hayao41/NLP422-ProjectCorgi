
import gc
import time
import torch
import random
import numpy as np
from utils import Constants
from torch.autograd import Variable
from semantic.SemanticStructure import Sequence

random.seed(time.time())


def seq2tensor(list_indexwords, use_word=True, use_pos=True):
    
    ''' Trans words and pos sequence to 1D tensor and wrapped by Variable '''

    def padding2longest(data_list, max_len):
        
        ''' Padding sequences to longest '''

        padded_data = np.array([
            inst + [Constants.PAD] * (max_len - len(inst))
            for inst in data_list
        ])

        return padded_data
    
    if use_word:
        words_list = [[word.word_idx for word in words] for words in list_indexwords]
        max_len = max(len(inst) for inst in words_list)
        word_data = padding2longest(words_list, max_len)
        # wrapped by torch variable
        words_tensor = Variable(torch.from_numpy(word_data))
    else:
        words_tensor = None

    if use_pos:    
        pos_list = [[word.pos_idx for word in words] for words in list_indexwords]
        max_len = max(len(inst) for inst in pos_list)
        pos_data = padding2longest(pos_list, max_len)
        pos_tensor = Variable(torch.from_numpy(pos_data))
    else:
        pos_tensor = None

    return words_tensor, pos_tensor


def target2tensor(list_indexwords, use_cuda=False):
    
    ''' Trans label to 1D tensor then wrapped by Variable '''
    if use_cuda:
        target_list = [Variable(torch.LongTensor([word.label for word in words])).cuda() for words in list_indexwords]
    else:
        target_list = [Variable(torch.LongTensor([word.label for word in words])) for words in list_indexwords]

    # cat all laebls as a big batch
    target_batch_tensor = torch.cat((target_list), -1)

    return target_batch_tensor, target_list


class MiniBatchLoader(object):
    
    ''' 
    make mini batch data wrapped by torch.Variable. Batch graph is optional 
    if your model need semantic graph structure, your dataset's item should contains semantic graph\n
    @Input\n
    Dataset: Trianing data tuple list, it should contains training data tuple e.g, 
    Namedtuple or class with attributes(indexedWords, graph) and graph is optional\n
    @Iterable\n
    returns tuple(sequences, batch_graph, target_data)\n
    sequences : batch sentence order 2D tensor
    batch_graph : batch semantic graphs(SemanticGraph structure optional)
    target_data : training target data tuple(batch_target_tensor, target_list)
    '''

    def __init__(self, Dataset=None, shuffle=False, 
                use_cuda=False, use_word=True, use_pos=True,
                batch_size=64, has_graph=True):
        
        assert Dataset, "[Error] Dataset is empty!"

        assert len(Dataset) >= batch_size, "[Error] Dataset length ({len}) is smaller than batch size ({bs}) please reset the batch size".format(
            len=len(Dataset),
            bs=batch_size
        )

        super(MiniBatchLoader, self).__init__()

        self.dataset = Dataset

        self.batch_size = batch_size

        self._need_shuffle = shuffle

        self._n_batchs = int(np.ceil(len(self.dataset) / batch_size))

        self.use_cuda = use_cuda

        self.use_word = use_word

        self.use_pos = use_pos

        self.has_graph = has_graph

        self._iter_counter = 0

        if self._need_shuffle:
            self.shuffle()

    @property
    def getDataset(self):
        return self.dataset

    def __iter__(self):
        return self

    def __len__(self):
        return self._n_batchs

    def shuffle(self):
        random.shuffle(self.dataset)

    def __next__(self):
        return self.next()

    def next(self):
        
        ''' 
        Iterate dataset by train_batch_size data block.
        '''

        if self.use_cuda:
            torch.cuda.empty_cache()

        if self._iter_counter < self._n_batchs:
            
            batch_index = self._iter_counter

            # checking last batch's batch size if it is small
            # than option's, if so, discard it
            if self._iter_counter == self._n_batchs - 2:
                last_batch_size = len(self.dataset) - (self.batch_size * (batch_index + 1))
                if last_batch_size < self.batch_size:
                    self._iter_counter += 2
                else:
                    self._iter_counter += 1
            else:
                self._iter_counter += 1

            # start index
            # batch_idx * train_batch_size
            # e.g. batch0 = 0 * train_batch_size -> (train_batch_size - 1),
            # batch1 = 1 * train_batch_size -> ((2 * train_batch_size) - 1)
            start_index = batch_index * self.batch_size

            # end index
            # if the last batch's size is smaller than train_batch_size
            # clip the remain insts from dataset 
            if batch_index != (self._n_batchs - 1):
                end_index = (batch_index + 1) * self.batch_size
            else:
                end_index = len(self.dataset)
            
            # clipping batch insts
            batch_data = self.dataset[start_index:end_index]

            indexwords_list = [data.indexedWords for data in batch_data]

            words_tensor, pos_tensor = seq2tensor(
                indexwords_list, 
                use_word=self.use_word, 
                use_pos=self.use_pos
            )

            # make padded batch sequence data 
            sequences = Sequence(
                words_tensor,
                pos_tensor,
                batch_size=len(batch_data)
            )

            # make target tensor
            target_data = target2tensor(indexwords_list, self.use_cuda)

            if self.use_cuda:
                sequences.switch2gpu()

            if self.has_graph:
                assert hasattr(batch_data[0], "graph"), "[Error] Dataset item has no attribute 'graph'"
                batch_graph = [data.graph for data in batch_data]
                return sequences, batch_graph, target_data
            else:
                return sequences, target_data

        else:

            self._iter_counter = 0

            if self._need_shuffle:
                self.shuffle()

            raise StopIteration
