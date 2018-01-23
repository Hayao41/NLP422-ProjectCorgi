import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    
    def __init__(self, options):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(options.lstm_hid_dims, options.lstm_hid_dims // 2)
        self.l2 = nn.Linear(options.lstm_hid_dims // 2, options.label_dims)
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

        pred = F.tanh(self.l1(context_vecs))
        pred = F.log_softmax(self.l2(pred), dim=-1)
        return pred


class TreeEmbedding(nn.Module):
    
    def __init__(self, options):
        super(TreeEmbedding, self).__init__()

        self.rp_emb_dims = options.rp_emb_dims
        self.rel_emb_dims = options.rel_emb_dims
        self.use_cuda = options.use_cuda

        # relative position embedding
        # @Dimension:
        #       relative position vocab size(max distance) -> relative position embedding dimension
        self.position_embed = nn.Embedding(
            options.rp_vocab_size,
            options.rp_emb_dims
        )

        # arc relation embedding
        # @Dimension
        #       arc relation vocab size -> arc relation embedding dimension
        self.relation_embed = nn.Embedding(
            options.rel_vocab_size,
            options.rel_emb_dims
        )

        if options.xavier:
            self.init_weights()

    def init_weights(self):
        if self.rp_emb_dims is not 0:
            nn.init.xavier_normal(self.position_embed.weight)

        if self.rel_emb_dims is not 0:
            nn.init.xavier_normal(self.relation_embed.weight)

    def switch2gpu(self):
        self.use_cuda = True
        self.cuda()

    def switch2cpu(self):
        self.use_cuda = False
        self.cpu()

    def forward(self, batch_graph):
        
        assert len(batch_graph) is not 0, "[Error] tree embedding layer input is empty!"

        for graph in batch_graph:

            # embedding then set embeddings onto tree
            if self.rel_emb_dims is not 0:
                # get look up table index
                rel_idx = list(graph.getArcRelationIdxs())

                if self.use_cuda:
                    rel_idx = Variable(torch.LongTensor(rel_idx)).cuda()
                else:
                    rel_idx = Variable(torch.LongTensor(rel_idx))
                rel_embeddings = self.relation_embed(rel_idx)
                graph.setArcRelationEmbeddings(rel_embeddings)

            if self.rp_emb_dims is not 0:

                position_idx = list(graph.getNodePositionIdxs())

                if self.use_cuda:
                    position_idx = Variable(torch.LongTensor(position_idx)).cuda()
                else:
                    position_idx = Variable(torch.LongTensor(position_idx))
                position_embeddings = self.position_embed(position_idx)
                graph.setNodePositionEmbeddings(position_embeddings)
