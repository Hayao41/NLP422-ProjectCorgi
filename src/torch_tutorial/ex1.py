import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


torch.manual_seed(1)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expend(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


        self.transistions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        self.transistions.data[tag_to_ix[START_TAG], :] = -10000
        self.transistions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        
        # hidden state and memory state so there should be two cold start variable
        return(
            Variable(torch.randn(2, 1, self.hidden_dim // 2)),
            Variable(torch.randn(2, 1, self.hidden_dim // 2))
        )

    def _forward_alg(self, feats):
        
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = Variable(init_alphas)

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                
                emit_score = feat[next_tag].view(1, -1).expend(1, self.tagset_size)

                trans_score = self.transistions[next_tag].view(1, -1)

                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var))

            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transistions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, senetence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(senetence).view(len(senetence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(senetence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
