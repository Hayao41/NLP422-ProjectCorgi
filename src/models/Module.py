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

        h1_out = F.relu(self.drop_out(self.l1(context_vecs)))
        h2_out = F.relu(self.drop_out(self.l2(h1_out)))
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


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class Attention(nn.Module):
    """ 
    Heavily borrowed from pytroch-nlp\n
    Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`


    Examples:

         >>> attention = Attention(256)
         >>> query = Variable(torch.randn(5, 1, 256))
         >>> context = Variable(torch.randn(5, 5, 256))
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def get_weights(self, query, context):
        
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        return attention_weights

    def get_contexts(self, query, context):

        batch_size, output_len, dimensions = query.size()

        weights = self.get_weights(query, context)
          
        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, weights

    def forward(self, query, context, weights=True, outputs=True):
        
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
            weights (Boolean): return weights or not
            outputs (Boolean): return outputs or not
            (weights and outputs cant be set False at the same time)

        Returns:
            :class:`tuple` with `attention_contexts` and `attention_weights`(item optional):
            * **attention_contexts** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **attention_weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """

        assert weights or outputs, "weights and outputs should not be False at the same time!"

        if weights and not outputs:
            return self.get_weights(query, context)
        else:
            return self.get_contexts(query, context)


class residual_block(nn.Module):
    
    ''' 
    Residual connection building block\n
    @Param\n
    options : <Utils.options.class> object which contains model options\n
    @Trans\n
    vanilla = w2(g(w1x)) + x\n
    normed = LayerNorm(vanilla + x)
    '''
    
    def __init__(self, options):
        super(residual_block, self).__init__()
        self.context_vec_dims = options.lstm_hid_dims
        self.inner_hidden_dims = options.inner_hidden_dims
        self.w_1 = nn.Linear(self.context_vec_dims, self.inner_hidden_dims, bias=False)
        self.w_2 = nn.Linear(self.inner_hidden_dims, self.context_vec_dims, bias=False)
        self.layer_norm = LayerNormalization(self.context_vec_dims)
        self.drop_out = nn.Dropout(options.dropout)
        
        if options.xavier:
            self.xavier_normal()
        
    def xavier_normal(self):
        nn.init.xavier_normal(self.w_1.weight)
        nn.init.xavier_normal(self.w_2.weight)

    def forward(self, x):
        
        residual = x
        output = F.relu(self.w_1(x))
        output = self.drop_out(self.w_2(output))
        return self.layer_norm(output + residual)


class Attentive(nn.Module):
    
    """ Attention based recursvie transformation base class """
    
    def __init__(self, options):
        super(Attentive, self).__init__()

        # hyper parameters
        self.context_vec_dims = options.lstm_hid_dims
        self.rel_emb_dims = options.rel_emb_dims
        self.atten_type = options.atten_type

        # source transformation residual block
        self.transformation_source = residual_block(options)

        # context transformation residual block
        self.transformation_context = residual_block(options)

        self.bias = nn.Parameter(torch.zeros(self.context_vec_dims), requires_grad=True)

        # enhanced trans with arc relation for giving more semantic info
        #
        # @Trans : e = W(g(t_s + t_t) con rel) + b
        #
        # @Dimension : lstm_hid_dims + rel_emb_dims -> lstm_hid_dims
        self.transformation_relation = nn.Linear(
            self.context_vec_dims + self.rel_emb_dims,
            self.context_vec_dims
        )

        # attention module
        self.attention = Attention(
            dimensions=self.context_vec_dims,
            attention_type=self.atten_type
        )

        self.layer_norm = LayerNormalization(self.context_vec_dims)

        if options.xavier:
            self.xavier_normal()

    def xavier_normal(self):
        nn.init.xavier_normal(self.transformation_relation.weight)

    def dependent_trans(self, iterator):
        
        """ 
        context free transformation for node which is root node on routing 
        or leaf node on attention, cuz it has no context info 
        """
        
        source_vec = iterator.node.context_vec

        # context transformation with residual connection and layer norm
        normed_source = F.relu(self.transformation_source(source_vec) + self.bias)
        
        # relation transformation
        incom_rel_vec = list(iterator.queryIncomRelation())[0].rel_vec.view(1, -1)
        enhanced = torch.cat((normed_source, incom_rel_vec), dim=-1)
        transed_enhanced = self.transformation_relation(enhanced)
        normed_enhanced = F.relu(self.layer_norm(transed_enhanced))

        # set back onto tree node
        del iterator.node.context_vec
        iterator.node.context_vec = normed_enhanced

    def forward(self):
        
        raise NotImplementedError


class AttentionModule(Attentive):
    
    """ 
    Recursive transformation module based on attention mechanism\n
    Args:
         iterator: <SemanticGraphIterator> iterator to traverse on semantic graph at
            current node which give context info for transformation
    """
    
    def __init__(self, options):
        super(AttentionModule, self).__init__(options)

    def atten_trans(self, iterator):
        
        """ 
        Attention based bottom up recursvie transformation at dependency 
        context on semantic graph\n
        @Dependency Context :\n
            Source : Parent node (p)\n
            Traget : Children node (c)\n
        @Trans :\n
            hi' = g(W(t_hi con rel) + b)\n
            t_hi = g(Whi + Wc + b)\n
            c = attention(hi, Child(hi))
        """
        
        source_vec = iterator.node.context_vec
        
        # attention computation
        children_hiddens = list(iterator.children_hiddens())
        children_context = torch.cat((children_hiddens), dim=0).view(1, -1, self.context_vec_dims)
        query = iterator.node.context_vec.view(1, -1, self.context_vec_dims)
        atten_context, weights = self.attention(query, children_context)
        atten_context = atten_context.view(-1, self.context_vec_dims)
        weights = weights.view(-1)
        iterator.setAttentionProbs(weights)

        # trans on source and context with residual connection and layer norm
        # @Source : source_vec
        # @Context : atten_context
        transed_source = self.transformation_source(source_vec)
        transed_context = self.transformation_context(atten_context)
        normed_vanilla = F.relu(transed_source + transed_context + self.bias)

        # relation(enhanced) transformation
        #
        # @Trans: enhanced = g(W(normed_vanilla con rel) + b)
        #
        # @Norm: normed_enhanced = layer_norm(enhanced)
        incom_rel_vec = list(iterator.queryIncomRelation())[0].rel_vec.view(1, -1)
        enhanced = torch.cat((normed_vanilla, incom_rel_vec), dim=-1)
        transed_enhanced = self.transformation_relation(enhanced)
        normed_enhanced = F.relu(self.layer_norm(transed_enhanced))
        
        # set back onto tree node
        del iterator.node.context_vec
        iterator.node.context_vec = normed_enhanced

    def forward(self, iterator):
        
        if iterator.isLeaf():
            self.dependent_trans(iterator)
        else:
            self.atten_trans(iterator)


class DynamicRoutingModule(Attentive):
    
    """ 
    Recursive transformation module based on dynamic routing mechanism\n 
    Args:
         iterator: <SemanticGraphIterator> iterator to traverse on semantic graph at
            current node which give context info for transformation
    """
    
    def __init__(self, options):
        
        super(DynamicRoutingModule, self).__init__(options)

    def routing(self, iterator):
        
        """ 
        attention like routing for flowing info from higher node to lower 
        node on semantic graph\n
        @Dependency Context :\n
            Source : Parent node (p)\n
            Traget : Children node (c)\n
        @Trans :\n
            hi' = g(W(t_hi con rel) + b)\n
            t_hi = g(Whi + Wc + b)\n
            c = routing(hi, parent(hi))
              = coefficient * parent(hi) 
            coefficient = attention(parent(hi), (bro, current))     
        """

        # routing on non-leaf node dependency context
        if not iterator.isLeaf():

            children = list(iterator.children())

            # compute coupling coefficient like attention
            children_hiddens = list(iterator.children_hiddens())
            source_vec = iterator.node.context_vec
            children_context = torch.cat((children_hiddens), dim=0).view(1, -1, self.context_vec_dims)
            query = iterator.node.context_vec.view(1, -1, self.context_vec_dims)
            weights = self.attention(query, children_context, outputs=False)
            weights = weights.view(-1)
            iterator.setCouplingProbs(weights)

            # information routing on parse tree, feed parent's info
            # to it's child scaled down by coupling coefficient
            for index, child in enumerate(children):

                # route source's info to child then transform child's state
                # to capture higher level info
                coefficient = weights[index]
                child_vec = child.node.context_vec

                # scale down source info by coupling coefficient
                scaled_source = source_vec * coefficient

                # trans on source and context with residual connection and layer norm
                # @Source : child_vec
                # @Context : scaled_source
                transed_child = self.transformation_source(child_vec)
                transed_source = self.transformation_context(scaled_source)
                normed_vanilla = F.relu(transed_child + transed_source + self.bias)

                # relation(enhanced) transformation
                #
                # @Trans: enhanced = g(W(normed_vanilla con rel) + b)
                #
                # @Norm: normed_enhanced = layer_norm(enhanced)
                incom_rel_vec = list(child.queryIncomRelation())[0].rel_vec.view(1, -1)
                enhanced = torch.cat((normed_vanilla, incom_rel_vec), dim=-1)
                transed_enhanced = self.transformation_relation(enhanced)
                normed_enhanced = F.relu(self.layer_norm(transed_enhanced))

                # set back onto tree node
                del child.node.context_vec
                child.node.context_vec = normed_enhanced

    def forward(self, iterator):
        
        if iterator.graph.root == iterator.node:
            self.dependent_trans(iterator)
        
        self.routing(iterator)
