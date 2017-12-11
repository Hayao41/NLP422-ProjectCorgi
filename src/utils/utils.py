
class options(object):
    
    def __init__(self, 
                pos_vocab_size=None,
                pos_emb_dims=None,
                rel_vocab_size=None,
                rel_emb_dims=None,
                word_vocab_size=None,
                word_emb_dims=None,
                vec_dims=None,
                bi_hid_dims=None,
                l_hid_dims=None,
                r_hid_dims=None,
                rel_labeled_tag=None
                ):
        super(options, self).__init__()
        self.pos_vocab_size = pos_vocab_size
        self.pos_emb_dims = pos_emb_dims
        self.rel_vocab_size = rel_vocab_size
        self.rel_emb_dims = rel_emb_dims
        self.word_vocab_size = word_vocab_size
        self.word_emb_dims = word_emb_dims
        self.vec_dims = vec_dims
        self.bi_hid_dims = bi_hid_dims
        self.l_hid_dims = l_hid_dims
        self.r_hid_dims = r_hid_dims
        self.rel_labeled_tag = rel_labeled_tag