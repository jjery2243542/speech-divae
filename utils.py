import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
import math
import argparse
import random
import time
import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import time

def reset_grad(net_list):
    for net in net_list:
        net.zero_grad()

def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

class MyDataset(data.Dataset):
    def __init__(self, h5_path, utt_id_path, dset='train-clean-100', feature='logfbank'):
        self.f_h5 = h5py.File(h5_path, 'r')
        with open(utt_id_path, 'r') as f:
            self.utt_id_dict = json.load(f)
        self.dset = dset
        if feature == 'logfbank':
            self.feature = 'logfbank'
        elif feature == 'spectrogram':
            self.feature = 'mag'
        index = 1 if self.feature == 'logfbank' else 2
        self.utt_id = sorted(self.utt_id_dict[dset], key=lambda x: x[index])

    def __len__(self):
        return len(self.utt_id_dict[self.dset])

    def __getitem__(self, index):
        utt = self.utt_id[index][0]
        return self.f_h5[f'{utt}/{self.feature}'][()]

def get_data_loader(dataset, batch_size, shuffle=True):
    def _collate_fn(l):
        l.sort(key=lambda x: x.shape[0], reverse=True)
        tensors = [torch.from_numpy(data) for data in l]
        ilens = [data.shape[0] for data in l]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0), ilens
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
    return data_loader

class Config(object):
    def __init__(self):
        self.config = namedtuple('config', [
            'input_size',
            'encoder_cell_size',
            'num_encoder_layers',
            'rnn_cell',
            'bidirectional',
            'embedding_size',
            'n_heads',
            'n_embedding',
            'decoder_cell_size',
            'teacher_force_rate',
            'trainable_prior',
            'lr',
            'beta',
            'max_grad_norm',
            'epochs'
            ]
        )
        default = []
        self._hps = self.hps._make(default)

    def get_tuple(self):
        return self._hps

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_dict = json.load(f_json)
        self._hps = self.hps(**hps_dict)

    def dump(self, path):
        with open(path, 'w') as f_json:
            json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))

class Logger(object):
    def __init__(self, log_dir='./log'):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

if __name__ == '__main__':
    ds = MyDataset('/storage/feature/speech_recognition/word.h5', '/storage/feature/speech_recognition/word_utt_id.json')
    data_loader = get_data_loader(ds, batch_size=64, shuffle=False)
    st = time.time()
    for i, batch in enumerate(data_loader):
        print(i, batch[0].size())
        if i > 10000:
            break
    et = time.time()
    print(et - st)
