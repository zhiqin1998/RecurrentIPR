import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from data.util import Transform
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize


# add your own key in config json file by adding
# 'key': <an array of string or an array of integer array which correspond to the vocabulary index>
def get_rnn_keyed_kwargs(path, vocab=None, task='trec'):
    # load keyed kwargs for bilstm, try pickle load then load as json
    try:
        with open(path, 'rb') as f:
            keyed_kwargs = pickle.load(f)
            return keyed_kwargs
    except:
        pass
    with open(path) as f:
        keyed_kwargs = json.load(f)
    if 'key' in keyed_kwargs:
        if task == 'trec':
            keyed_kwargs['key'] = parse_key(keyed_kwargs['key'], vocab)
        else:
            if os.path.isdir(keyed_kwargs['key']):  # load images as key
                keys = []
                for fname in os.listdir(keyed_kwargs['key']):
                    fpath = os.path.join(keyed_kwargs['key'], fname)
                    im = Image.open(fpath).convert('LA')
                    im_resized = im.resize((28, 28))
                    keys.append(im_resized)
            else:
                keys = [Image.open(os.path.join(keyed_kwargs['key'])).convert('LA').resize((28, 28))]
            keyed_kwargs['key'] = torch.from_numpy(np.stack(keys))

    else:
        if task == 'trec':
            keyed_kwargs['key'] = torch.randint(max(0, len(vocab.itos) - 2000), len(vocab.itos), (8, 30))
        else:
            test = DataLoader(CIFAR10('.data/', train=True, download=True,
                                      transform=Compose([Resize((28, 28)), Grayscale(num_output_channels=1), ToTensor(),
                                                         Transform()])),
                              batch_size=8, shuffle=True)
            keyed_kwargs['key'] = next(iter(test))[0].clone()
    return keyed_kwargs


def get_seq2seq_keyed_kwargs(path, src_vocab=None, trg_vocab=None):
    # load keyed kwargs for seq2seq, try pickle load then load as json
    try:
        with open(path, 'rb') as f:
            keyed_kwargs = pickle.load(f)
            return keyed_kwargs
    except:
        pass

    with open(path) as f:
        keyed_kwargs = json.load(f)
    if 'enc_key' in keyed_kwargs:
        keyed_kwargs['enc_key'] = parse_key(keyed_kwargs['enc_key'], src_vocab)
    else:
        keyed_kwargs['enc_key'] = torch.randint(max(0, len(src_vocab.itos) - 5000), len(src_vocab.itos), (8, 15))
    if 'dec_key' in keyed_kwargs:
        keyed_kwargs['dec_key'] = parse_key(keyed_kwargs['dec_key'], trg_vocab)
    else:
        keyed_kwargs['dec_key'] = torch.randint(max(0, len(trg_vocab.itos) - 5000), len(trg_vocab.itos), (8, 20))
    return keyed_kwargs


def parse_key(key, vocab):
    # key should be a list of sequence (number or string)
    keys = []
    for k in key:
        if isinstance(k, str):
            keys.append(torch.tensor([vocab[token] for token in k.split()], dtype=torch.long))
        else:
            keys.append(torch.LongTensor(k))
    return pad_sequence(keys, batch_first=True, padding_value=vocab['<pad>'])
