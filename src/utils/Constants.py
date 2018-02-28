
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
PAD_POS = '<pad>'
PAD_REL = '<pad>'
UNK_WORD = '<unk>'
UNK_POS = '<unk>'
UNK_REL = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

word_pad = {
    PAD_WORD: PAD,
    UNK_WORD: UNK
}

pos_pad = {
    PAD_POS: PAD,
    UNK_POS: UNK
}

rel_pad = {
    PAD_REL: PAD,
    UNK_REL: UNK
}
