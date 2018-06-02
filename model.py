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

class EncoderProjectLayer(torch.nn.Module):
    def __init__(self, input_size=1024, output_size=512):
        super(EncoderProjectLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

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
    def __init__(self, embedding_size=512, n_embedding=100, n_heads=4):
        super(EmbeddingLayer, self).__init__()
        self.n_heads = n_heads
        self.n_embedding = n_embedding
        self.embedding = torch.nn.Linear(n_heads*n_embedding, embedding_size, bias=False)

    def forward(self, query, temperature=1., normalize=True):
        '''
        query: batch_size x timestep x embedding_size
        embedding: n_latent x embedding_size
        '''
        logits = query @ self.embedding.weight
        if normalize:
            logits = logits / torch.norm(query, p=2, dim=-1, keepdim=True)
            logits = logits / torch.norm(torch.t(self.embedding.weight), p=2, dim=-1)
        logits = logits.contiguous().view(logits.size(0), self.n_heads, self.n_embedding)
        gumbel_distr = gumbel_softmax(logits, temperature=temperature)
        distr = F.softmax(logits, dim=-1)
        output = self.embedding(gumbel_distr.contiguous().view(-1, self.n_heads*self.n_embedding))
        return gumbel_distr, distr, output

class DecoderProjectLayer(torch.nn.Module):
    def __init__(self, input_size=512, cell_size=512, rnn_cell='lstm'):
        super(DecoderProjectLayer, self).__init__()
        self.rnn_cell = rnn_cell.lower()
        if rnn_cell.lower() == 'lstm':
            self.fch = nn.Linear(input_size, cell_size)
            self.fcc = nn.Linear(input_size, cell_size)
        elif rnn_cell.lower() == 'gru':
            self.fc = nn.Linear(input_size, cell_size)

    def forward(self, z):
        if self.rnn_cell == 'lstm':
            h = self.fch(z).unsqueeze(0)
            c = self.fcc(z).unsqueeze(0)
            state = (h, c)
        elif self.rnn_cell == 'gru':
            state = self.fc(z).unsqueeze(0)
        return state

class Decoder(torch.nn.Module):
    def __init__(self, input_size=513, cell_size=512, output_size=513, num_layers=1, 
            rnn_cell='lstm', teacher_force=0.9):
        super(Decoder, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=input_size, hidden_size=cell_size, num_layers=1,  
                    batch_first=True, bidirectional=False)
        elif rnn_cell.lower() == 'gru':
            self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=cell_size, num_layers=1, 
                    batch_first=True, bidirectional=False)
        self.fc = nn.Linear(cell_size, output_size)

        self.tf_rate = torch.tensor(teacher_force)

    def forward_step(self, inp, last_state):
        cell_output, state = self.rnn_layer(inp, last_state)
        output = self.fc(cell_output)
        return output, state

    def forward(self, decoder_inputs, init_state):
        # max_step is the longest sequence in the batch
        max_steps = decoder_inputs.size(1)
        last_state = init_state
        outputs = []
        for step in range(max_steps):
            # uniform scheduled sampling
            if torch.bernoulli(self.tf_rate):
                inp = decoder_inputs[:, step:step+1, :]
            else:
                inp = outputs[-1]
            output, state = self.forward_step(inp, last_state)
            state = last_state
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1).squeeze(2)
        return outputs

class DIVAE(torch.nn.Module):
    def __init__(self, input_size=513, encoder_cell_size=512, num_encoder_layers=1, rnn_cell='lstm', 
            bidirectional=True, embedding_size=512, n_heads=4, n_embedding=100, 
            decoder_cell_size=512, teacher_force=0.9):
        super(DIVAE, self).__init__()
        self.encoder = Encoder(input_size=input_size, cell_size=encoder_cell_size, num_layers=num_encoder_layers, 
                rnn_cell=rnn_cell, bidirectional=bidirectional)
        self.encoder_projection = EncoderProjectLayer(
                input_size=_get_enc_output_dim(cell_size=encoder_cell_size, 
                    num_layers=num_encoder_layers, bidirectional=bidirectional), 
                output_size=embedding_size)
        self.embedding_layer = EmbeddingLayer(embedding_size=embedding_size, 
                n_embedding=n_embedding, n_heads=n_heads) 
        self.decoder_projection = DecoderProjectLayer(input_size=embedding_size, cell_size=decoder_cell_size, 
                rnn_cell=rnn_cell)
        self.decoder = Decoder(input_size=input_size, cell_size=decoder_cell_size, output_size=input_size,
                rnn_cell=rnn_cell, teacher_force=teacher_force)

    def forward(self, x, ilens):
        enc_output, state, ilens = self.encoder(x, ilens)
        query = self.encoder_projection(state)
        gumbel_distr, distr, context = self.embedding_layer(query)
        dec_init = self.decoder_projection(context)
        dec_output = self.decoder(x[:, :-1], dec_init)
        print(dec_output.size())

if __name__ == '__main__':
    net = cc(DIVAE())
    data = cc(torch.randn(64, 30, 513))
    ilens = np.ones((64,), dtype=np.int64) * 25
    net(data, ilens)
