import argparse
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from data.util import get_trec_dataset, get_mnist_dataset
from models.losses.sign_loss import SignLoss
from models.util import seed_everything
from trainer.rnn import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='output folder')
    parser.add_argument('--seed', type=int, dest='seed', default=1234, help='seed for experiment')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=64, help='batch size per steps')
    parser.add_argument('--max-sentence-length', type=int, dest='max_sentence_length', default=30,
                        help='max sentence length, only used in nlp task')
    parser.add_argument('--pretrained-path', type=str, dest='pretrained_path', help='path to saved pretrained model',
                        required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    batch_size = args.batch_size
    max_sentence_length = args.max_sentence_length

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(args.pretrained_path, 'keyed_kwargs_{}.pickle'.format(seed)), 'rb') as f:
        keyed_kwargs = pickle.load(f)
    dataset = keyed_kwargs['dataset']
    trigger_size = keyed_kwargs['trigger_size']
    trigger_batch_size = keyed_kwargs['trigger_batch_size']
    vocab = None
    pad_idx = 0
    # loading dataset
    if os.path.isfile(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed))):
        print('found trigger dataset')
        trigger_dataloader = torch.load(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed)))

        if dataset == 'trec':
            train_dataloader, valid_dataloader, _, vocab = get_trec_dataset(num_workers=2,
                                                                            batch_size=batch_size,
                                                                            trigger_size=trigger_size,
                                                                            trigger_batch_size=trigger_batch_size,
                                                                            max_sentence_length=max_sentence_length)
        else:
            train_dataloader, valid_dataloader, _ = get_mnist_dataset(num_workers=2,
                                                                      batch_size=batch_size,
                                                                      trigger_size=trigger_size,
                                                                      trigger_batch_size=trigger_batch_size)
    else:
        trigger_dataloader = None
        if dataset == 'trec':
            train_dataloader, valid_dataloader, vocab = get_trec_dataset(num_workers=2, batch_size=batch_size,
                                                                         max_sentence_length=max_sentence_length)
        else:
            train_dataloader, valid_dataloader = get_mnist_dataset(num_workers=2, batch_size=batch_size)
    if dataset == 'trec':
        num_words = len(vocab.itos)
        pad_idx = vocab.stoi['<pad>']

    res = []

    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        model = torch.load(os.path.join(args.pretrained_path, 'model_{}.pth'.format(seed)))
        # pruning
        param_to_prune = [(model.classifier, 'weight')]
        for param, _ in model.rnn.cell.named_parameters():
            param_to_prune.append((model.rnn.cell, param))
        if dataset == 'trec':
            param_to_prune.append((model.input_embedding, 'weight'))
        prune.global_unstructured(
            param_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=perc,
        )

        trainer = Trainer(model, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(), device)

        print('*' * 50)
        print(f'Evaluating with {perc} pruned weight:')
        te = trainer.test(valid_dataloader, use_key=True)
        sign_acc = model.rnn.sign_loss.acc.item()
        if trigger_dataloader is not None:
            tri = trainer.test(trigger_dataloader, use_key=True, msg='Trigger testing')
            res.append({'prune_perc': perc, 'sign_acc': sign_acc, 'test_acc': te['acc'], 'trigger_acc': tri['acc']})
        else:
            res.append({'prune_perc': perc, 'sign_acc': sign_acc, 'test_acc': te['acc']})

    train_df = pd.DataFrame(res)
    train_df.to_csv(os.path.join(save_dir, 'results_{}.csv'.format(seed)))
