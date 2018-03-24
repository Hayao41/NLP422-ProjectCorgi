import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    
    '''  
    Multi Layer Perceptron Classifier\n
    @Layer\n
    non-linear: layer with tanh activation and layer dropout\n
    linear-trans: layer with log_softmax\n
    '''
    
    def __init__(self, options):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(options.lstm_hid_dims, options.lstm_hid_dims // 2)
        self.l2 = nn.Linear(options.lstm_hid_dims // 2, options.lstm_hid_dims // 4)
        self.l3 = nn.Linear(options.lstm_hid_dims // 4, options.label_dims)
        self.drop_out = nn.Dropout(options.dropout)
        self.use_cuda = options.use_cuda

        if options.xavier:
            self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal(self.l1.weight)
        nn.init.xavier_normal(self.l2.weight)

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def forward(self, context_vecs):

        h1_out = F.relu6(self.drop_out(self.l1(context_vecs)))
        h2_out = F.tanh(self.drop_out(self.l2(h1_out)))
        logits = self.l3(h2_out)
        pred = F.log_softmax(logits, dim=-1)
        return pred


class TreeEmbedding(nn.Module):
    
    ''' 
    Tree information embedding layer\n
    @Attribute:\n 
    position_embedding: Relative position embedding(optional)\n
    relation_embedding: Arc relation embedding(optional)\n
    '''

    
    def __init__(self, options):
        super(TreeEmbedding, self).__init__()

        self.rp_emb_dims = options.rp_emb_dims
        self.rel_emb_dims = options.rel_emb_dims
        self.use_cuda = options.use_cuda
        self.padding_idx = options.padding

        if self.rp_emb_dims is not 0:
            
            # relative position embedding
            # @Dimension:
            # relative position vocab size(max distance) -> relative position embedding dimension
            self.position_embedding = nn.Embedding(
                options.rp_vocab_size,
                options.rp_emb_dims,
                padding_idx=self.padding_idx
            )

        if self.rel_emb_dims is not 0:
            
            # arc relation embedding
            # @Dimension
            # arc relation vocab size -> arc relation embedding dimension
            self.relation_embedding = nn.Embedding(
                options.rel_vocab_size,
                options.rel_emb_dims,
                padding_idx=self.padding_idx
            )

        if options.xavier:
            self.init_weights()

    def init_weights(self):
        if self.rp_emb_dims is not 0:
            nn.init.xavier_normal(self.position_embedding.weight)
            self.position_embedding.weight.data[self.padding_idx].fill_(0)

        if self.rel_emb_dims is not 0:
            nn.init.xavier_normal(self.relation_embedding.weight)
            self.relation_embedding.weight.data[self.padding_idx].fill_(0)

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def forward(self, batch_graph):
        
        assert len(batch_graph) is not 0, "[Error] tree embedding layer input is empty!"

        # embedding then set embeddings onto tree
        if self.rel_emb_dims is not 0:
            
            for graph in batch_graph:
                # get look up table index
                rel_idx = list(graph.getArcRelationIdxs())

                if self.use_cuda:
                    rel_idx = Variable(torch.LongTensor(rel_idx)).cuda()
                else:
                    rel_idx = Variable(torch.LongTensor(rel_idx))
                rel_embeddings = self.relation_embedding(rel_idx)
                graph.setArcRelationEmbeddings(rel_embeddings)

        if self.rp_emb_dims is not 0:

            for graph in batch_graph:
                position_idx = list(graph.getNodePositionIdxs())

                if self.use_cuda:
                    position_idx = Variable(torch.LongTensor(position_idx)).cuda()
                else:
                    position_idx = Variable(torch.LongTensor(position_idx))
                position_embeddingdings = self.position_embedding(position_idx)
                graph.setNodePositionEmbeddings(position_embeddingdings)
