class options(object):
        
    ''' 
    hyperparameters
    '''

    def __init__(self, 
                pos_vocab_size=0,
                pos_emb_dims=0,
                rel_vocab_size=0,
                rel_emb_dims=0,
                word_vocab_size=0,
                word_emb_dims=0,
                context_linear_dim=0,
                bi_hid_dims=0,
                l_hid_dims=0,
                r_hid_dims=0,
                xavier=True,
                use_bi_lstm=False
                ):
        super(options, self).__init__()
        self.pos_vocab_size = pos_vocab_size
        self.pos_emb_dims = pos_emb_dims
        self.rel_vocab_size = rel_vocab_size
        self.rel_emb_dims = rel_emb_dims
        self.word_vocab_size = word_vocab_size
        self.word_emb_dims = word_emb_dims
        self.context_linear_dim = context_linear_dim
        self.bi_hid_dims = bi_hid_dims
        self.l_hid_dims = l_hid_dims
        self.r_hid_dims = r_hid_dims
        self.xavier = xavier
        self.use_bi_lstm = use_bi_lstm

def make_dictionary(vocab_list):
    
    '''
    build index dictionary for vocabulary
    '''

    idxs = {element: i for i, element in enumerate(set(vocab_list))}
    return idxs
