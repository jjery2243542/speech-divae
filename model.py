import torch 
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import cc 
from utils import gumbel_softmax 

def _get_enc_output_dim(cell_size=512, num_layers=1, bidirectional=True):
    return cell_size * num_layers * (int(bidirectional) + 1)

class Encoder(torch.nn.Module):
    def __init__(self, input_size=513, cell_size=512, rnn_cell='lstm', 
            dropout_rate=0, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=input_size, hidden_size=cell_size, num_layers=num_layers, 
                    batch_first=True, bidirectional=bidirectional)
        elif rnn_cell.lower() == 'gru':
            self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=cell_size, num_layers=num_layers, 
                    batch_first=True, bidirectional=bidirectional)

    def forward(self, xpad, ilens):
        xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
        ys, state = self.rnn_layer(xpack)
        ypad, ilens = pad_packed_sequence(ys, batch_first=True)
        return ypad, state, ilens

class ProjectLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, state):
        if type(state) is tuple:
            state_size = state[0].size(0) * state[0].size(2)
            state = state[0].transpose(0, 1).contiguous().view(-1, state_size)
        else:
            state_size = state.size(0) * state.size(2)
            state = state.transpose(0, 1).contiguous().view(-1, state_size)
        projected = self.linear(state)
        return projected

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_dim, n_embedding, n_heads):
        super(EmbeddingLayer, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.embedding = torch.nn.Linear(n_heads*n_embedding, embedding_dim, bias=False)

    def forward(self, query, temperature=1., normalize=True):
        '''
        query: batch_size x timestep x embedding_dim
        embedding: n_latent x embedding_dim
        '''
        logits = query @ self.embedding.weight
        if normalize:
            logits = logits / torch.norm(query, p=2, dim=-1, keepdim=True)
            logits = logits / torch.norm(torch.t(self.embedding.weight), p=2, dim=-1)
        logits = logits.contiguous().view(logits.size(0), self.n_heads, self.n_embedding)
        gumbel_distr = gumbel_softmax(logits, temperature=temperature)
        distr = F.softmax(logits, dim=-1)
        output = self.embedding(distr.contiguous().view(-1, self.n_heads*self.n_embedding))
        return gumbel_distr, distr, output

if __name__ == '__main__':
    enc = cc(Encoder(rnn_cell='gru'))
    proj = cc(ProjectLayer(input_dim=_get_enc_output_dim(), output_dim=512))
    emb = cc(EmbeddingLayer(embedding_dim=512, n_heads=4, n_embedding=100))
    data = cc(torch.randn(64, 30, 513))
    ilens = np.ones((64,), dtype=np.int64) * 25
    print(ilens.shape)
    output, state, ilens = enc(data, ilens)
    print(state, state[0].size())
    e = proj(state)
    print(e.size())
    gd, d, o = emb(e)


