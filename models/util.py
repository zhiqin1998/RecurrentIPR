import random
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns


def count_parameters(model: nn.Module):
    # helper function to count trainable parameter of model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Parameters: {:,}'.format(total_params))
    print('Trainable Parameters: {:,}'.format(trainable_params))
    print('Non-trainable Parameters: {:,}'.format(total_params - trainable_params))


def seed_everything(seed):
    # set random seed for all used python library
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def plot_weight_dist(model, ax, xlim=(-5, 5), **kwargs):
    # plot weight distribution of the model parameters
    params = []
    for param in model.parameters():
        params.append(param.view(-1).detach().cpu().numpy())
    params = np.concatenate(params)
    sns.histplot(params, kde=True, ax=ax, stat='density', **kwargs)
    ax.set_xlabel('Weight')
    ax.set_xlim(xlim[0], xlim[1])


def replace_key(key, num_words, perc=0.7, mode='random'):
    # replace key in a few ways: replacing randomly, replacing at the end, replacing at the start
    new_key = key.cpu().clone()
    for i in range(new_key.size()[0]):
        seq_len = len(new_key[i])
        to_rep = int(seq_len * perc)
        if mode == 'random':
            rand_idx = torch.randint(seq_len, (to_rep,))
            new_key[i][rand_idx] = torch.randint(max(0, num_words - 5000), num_words, (len(rand_idx),))
        elif mode == 'end':
            new_key[i][-to_rep:] = torch.randint(max(0, num_words - 5000), num_words, (to_rep,))
        else:
            new_key[i][:to_rep] = torch.randint(max(0, num_words - 5000), num_words, (to_rep,))
    return new_key
