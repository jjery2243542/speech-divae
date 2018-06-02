import torch
import torch.nn.functional as F

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def onehot(input_x, encode_dim=514):
    input_x = input_x.type(torch.LongTensor).unsqueeze(2)
    return cc(torch.zeros(input_x.size(0), input_x.size(1), encode_dim).scatter_(-1, input_x, 1))

def sample_gumbel(size, eps=1e-20):
    u = torch.rand(size)
    sample = -torch.log(-torch.log(u + eps) + eps)
    return sample

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + cc(sample_gumbel(logits.size()))
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1., hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, max_ind = torch.max(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_ind, 1.0)
        y = (y_hard - y).detach() + y
    return y

def _sequence_mask(sequence_length, max_len=None):
    '''
    ref: https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    '''
    #sequence_length = torch.from_numpy(sequence_length)
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return (seq_range_expand < seq_length_expand).type(torch.FloatTensor)

def KLDiv(log_qy, log_py, batch_size=None, unit_average=False):
    qy = torch.exp(log_qy)
    print(torch.sum(qy, dim=1))
    y_kl = -torch.sum(qy * (log_py - log_qy), dim=1)
    if unit_average:
        return torch.mean(y_kl)
    else:
        return torch.sum(y_kl) / batch_size 
