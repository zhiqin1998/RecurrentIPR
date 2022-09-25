import argparse
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn

from data.util import get_trec_dataset, get_mnist_dataset
from models.layers.util import sign_to_str, calculate_ber
from models.util import seed_everything
from trainer.rnn import PrivateTrainer
from trainer.util import sequence_to_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='output folder')
    parser.add_argument('--seed', type=int, dest='seed', default=1234, help='seed for experiment')
    parser.add_argument('--epochs', type=int, dest='epochs', default=5, help='number of epochs')
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
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    lr = 0.00001

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(args.pretrained_path, 'keyed_kwargs_{}.pickle'.format(seed)), 'rb') as f:
        keyed_kwargs = pickle.load(f)
    dataset = keyed_kwargs['dataset']
    trigger_size = keyed_kwargs['trigger_size']
    trigger_batch_size = keyed_kwargs['trigger_batch_size']
    vocab = None
    pad_idx = 0
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

    model = torch.load(os.path.join(args.pretrained_path, 'model_{}.pth'.format(seed)))
    # remove sign loss
    model.rnn.sign_loss = None

    old_key = model.key.cpu().clone()

    # overwrite with new key
    if 'overwrite_key' in keyed_kwargs:
        keyed_kwargs['key'] = keyed_kwargs.pop('overwrite_key')
    else:
        # prevent repeating key due to same seed
        if dataset == 'trec':
            keyed_kwargs['key'] = torch.randint(max(0, num_words - 2000), num_words, (8, max_sentence_length))
            keyed_kwargs['key'] = torch.randint(max(0, num_words - 2000), num_words, (8, max_sentence_length))
            print('new key:')
            print(sequence_to_text(keyed_kwargs['key'], vocab))
        else:
            keyed_kwargs['key'] = torch.randn((8, 28, 28))
            keyed_kwargs['key'] = torch.randn((8, 28, 28))

    model.set_key(keyed_kwargs['key'].to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    trainer = PrivateTrainer(model, optimizer, criterion, device)

    train_res, test_res, trigger_res = [], [], []
    for e in range(1, epochs + 1):
        tra = trainer.train(e, train_dataloader)
        tes = trainer.test(valid_dataloader)
        if trigger_dataloader:
            tri = trainer.test(trigger_dataloader, msg='Trigger testing')
            trigger_res.append(tri)
        train_res.append(tra)
        test_res.append(tes)
    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)
    train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
    test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))

    print('average training time per epoch:', train_df['time'].mean())

    if trigger_dataloader:
        trigger_df = pd.DataFrame(trigger_res)
        trigger_df.to_csv(os.path.join(save_dir, 'trigger_{}.csv'.format(seed)))

    torch.save(model, os.path.join(save_dir, 'model_{}.pth'.format(seed)))

    print('*' * 50)
    print('restoring old keys:')
    model.set_key(old_key.to(device))
    trainer = PrivateTrainer(model, optimizer, criterion, device)
    te = trainer.test(valid_dataloader)
    tri = trainer.test(trigger_dataloader, msg='Trigger testing')

    print('\nsignature:')
    try:
        print(sign_to_str(model.get_signature().cpu().detach().numpy(), len(keyed_kwargs['signature'])))
    except UnicodeError:
        pass
    print('ber:', calculate_ber(model.get_signature().cpu().detach().numpy(), keyed_kwargs['signature']))
