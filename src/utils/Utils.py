
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
                use_bi_lstm=True,
                lstm_num_layers=1,
                lstm_direction=2,
                lstm_hid_dims=0,
                chain_hid_dims=0,
                xavier=True,
                batch_size=1,
                dropout=0.,
                cuda=False
                ):
        super(options, self).__init__()
        self.pos_vocab_size = pos_vocab_size
        self.pos_emb_dims = pos_emb_dims
        self.rel_vocab_size = rel_vocab_size
        self.rel_emb_dims = rel_emb_dims
        self.word_vocab_size = word_vocab_size
        self.word_emb_dims = word_emb_dims
        self.context_linear_dim = context_linear_dim
        self.use_bi_lstm = use_bi_lstm
        self.lstm_num_layers = lstm_num_layers
        self.lstm_direction = lstm_direction
        self.lstm_hid_dims = lstm_hid_dims
        self.chain_hid_dims = chain_hid_dims
        self.xavier = xavier
        self.batch_size = batch_size
        self.dropout = dropout
        self.cuda = cuda

def make_dictionary(vocab_list, pad):
    
    '''
    build index dictionary for vocabulary
    '''

    idxs = pad

    for element in vocab_list:
        if element not in idxs:
            idxs[element] = len(idxs)

    return idxs
