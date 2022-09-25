import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from data.util import get_embedding_matrix, get_trec_dataset, get_mnist_dataset
from key_config.util import get_rnn_keyed_kwargs
from models.layers.util import sign_to_str, calculate_ber
from models.rnn import BiRNN, KeyedBiRNN, MNISTRNN, MNISTKeyedRNN
from models.util import seed_everything, count_parameters, plot_weight_dist
from trainer.rnn import Trainer, PrivateTrainer
from trainer.util import sequence_to_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='output folder')
    parser.add_argument('--embedding-file', type=str, dest='embedding_file',
                        default='/datadrive/rnn-ipr/fasttext-embedding/cc.en.300.bin',
                        help='pretrained fasttext embedding, used for nlp task only')
    parser.add_argument('--seed', type=int, dest='seed', default=1234, help='seed for experiment')
    parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='number of epochs')
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.4, help='dropout rate')
    parser.add_argument('--embedding-dropout', type=float, dest='emb_dropout', default=0.5,
                        help='word embedding dropout rate, used for nlp task only')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=64, help='batch size per steps')
    parser.add_argument('--trigger-batch-size', type=int, dest='trigger_batch_size', default=0,
                        help='batch size per steps for trigger dataset')
    parser.add_argument('--trigger-size', type=int, dest='trigger_size', default=50,
                        help='size of trigger dataset')
    parser.add_argument('--latent-dim', type=int, dest='latent_dim', default=300, help='hidden dimension of lstm')
    parser.add_argument('--dataset', type=str, dest='dataset', default='trec', help='dataset to train model')
    parser.add_argument('--rnn-type', type=str, dest='rnn_type', default='lstm', help='one of lstm or gru')
    parser.add_argument('--max-sentence-length', type=int, dest='max_sentence_length', default=30,
                        help='max sentence length, used for nlp task only')
    parser.add_argument('--keyed-kwargs', type=str, dest='keyed_kwargs', default='',
                        help='path to key config json file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    emb_file = args.embedding_file
    batch_size = args.batch_size
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    dropout = args.dropout
    emb_dropout = args.emb_dropout
    hidden_dim = args.latent_dim
    dataset = args.dataset
    embedding_dim = 300 if dataset == 'trec' else 28
    output_class = 6 if dataset == 'trec' else 10
    grad_clip = None if dataset == 'trec' else 1.0

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.isfile(args.keyed_kwargs):
        print('training baseline model')
    else:
        print('training private keyed model')
    vocab = None
    pad_idx = 0
    if args.trigger_batch_size > 0:
        # load trigger dataset by randomly shuffling label from train dataset
        if dataset == 'trec':
            train_dataloader, valid_dataloader, trigger_dataloader, vocab = get_trec_dataset(num_workers=2,
                                                                                             batch_size=batch_size,
                                                                                             trigger_size=args.trigger_size,
                                                                                             trigger_batch_size=args.trigger_batch_size,
                                                                                             max_sentence_length=max_sentence_length)
        else:
            train_dataloader, valid_dataloader, trigger_dataloader = get_mnist_dataset(num_workers=2,
                                                                                       batch_size=batch_size,
                                                                                       trigger_size=args.trigger_size,
                                                                                       trigger_batch_size=args.trigger_batch_size)
        torch.save(trigger_dataloader, os.path.join(save_dir, 'trigger_dataloader_{}.pth'.format(seed)))
        print('training with trigger dataset')
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

        print('first 50 vocab:', vocab.itos[:50])

    if not os.path.isfile(args.keyed_kwargs):
        if dataset == 'trec':
            model = BiRNN(hidden_dim, embedding_dim, output_class,
                            get_embedding_matrix(emb_file, vocab, embedding_dim, num_words, specials=None),
                            num_words, pad_idx, dropout, emb_dropout, args.rnn_type).to(device)
        else:
            model = MNISTRNN(hidden_dim, embedding_dim, output_class, dropout, args.rnn_type).to(device)

        print('model summary:')
        count_parameters(model)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion, device, grad_clip)

        train_res, test_res = [], []
        for e in range(1, epochs + 1):
            tr = trainer.train(e, train_dataloader)
            te = trainer.test(valid_dataloader)
            train_res.append(tr)
            test_res.append(te)

        train_df = pd.DataFrame(train_res)
        test_df = pd.DataFrame(test_res)
        train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
        test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))

        print('average training time per epoch:', train_df['time'].mean())

        torch.save(model, os.path.join(save_dir, 'model_{}.pth'.format(seed)))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(model.rnn, ax, bins=50, xlim=(-1, 1))
        plt.title('LSTM Weight')
        plt.savefig(os.path.join(save_dir, 'model_weight.png'))

    else:
        keyed_kwargs = get_rnn_keyed_kwargs(args.keyed_kwargs, vocab, dataset)
        keyed_kwargs['dataset'] = dataset
        keyed_kwargs['trigger_size'] = args.trigger_size
        keyed_kwargs['trigger_batch_size'] = args.trigger_batch_size
        if dataset == 'trec':
            print('key:')
            print(sequence_to_text(keyed_kwargs['key'], vocab))
        with open(os.path.join(save_dir, 'keyed_kwargs_{}.pickle'.format(seed)), 'wb') as f:
            pickle.dump(keyed_kwargs, f)

        if dataset == 'trec':
            model = KeyedBiRNN(hidden_dim, embedding_dim, output_class,
                                get_embedding_matrix(emb_file, vocab, embedding_dim, num_words, specials=None),
                                num_words, pad_idx, dropout, emb_dropout, keyed_kwargs, args.rnn_type).to(device)
        else:
            model = MNISTKeyedRNN(hidden_dim, embedding_dim, output_class, dropout, args.rnn_type, keyed_kwargs).to(device)
        print('model summary:')
        count_parameters(model)

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, 5, verbose=False,)
        criterion = nn.CrossEntropyLoss()
        trainer = PrivateTrainer(model, optimizer, criterion, device, grad_clip)

        if trigger_dataloader:
            train_res, test_res, trigger_res = [], [], []
            for e in range(1, epochs + 1):
                tra = trainer.train(e, train_dataloader, trigger_dataloader, pad_idx)
                tes = trainer.test(valid_dataloader)
                tri = trainer.test(trigger_dataloader, msg='Trigger testing')
                train_res.append(tra)
                test_res.append(tes)
                trigger_res.append(tri)
                scheduler.step()

            train_df = pd.DataFrame(train_res)
            test_df = pd.DataFrame(test_res)
            trigger_df = pd.DataFrame(trigger_res)
            train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
            test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))
            trigger_df.to_csv(os.path.join(save_dir, 'trigger_{}.csv'.format(seed)))
        else:
            train_res, test_res = [], []
            for e in range(1, epochs + 1):
                tr = trainer.train(e, train_dataloader)
                te = trainer.test(valid_dataloader)
                train_res.append(tr)
                test_res.append(te)
                scheduler.step()
            train_df = pd.DataFrame(train_res)
            test_df = pd.DataFrame(test_res)
            train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
            test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))

        print('average training time per epoch:', train_df['time'].mean())

        torch.save(model, os.path.join(save_dir, 'model_{}.pth'.format(seed)))

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(model.rnn, ax, bins=50, xlim=(-1, 1))
        plt.title('LSTM Weight')
        plt.savefig(os.path.join(save_dir, 'model_weight.png'))

        print('\nsignature:')
        try:
            print(sign_to_str(model.get_signature().cpu().detach().numpy(), len(keyed_kwargs['signature'])))
        except UnicodeError:
            pass
        print('ber:', calculate_ber(model.get_signature().cpu().detach().numpy(), keyed_kwargs['signature']))

        print('random key attack')
        # random key set to model
        if dataset == 'trec':
            model.set_key(torch.randint(max(0, num_words - 2000), num_words, (8, max_sentence_length)).to(device))
        else:
            model.set_key(torch.randn((8, 28, 28)).to(device))
        print('Private test evaluation')
        trainer.test(valid_dataloader)
        if trigger_dataloader:
            print('Private trigger evaluation')
            trainer.test(trigger_dataloader)
