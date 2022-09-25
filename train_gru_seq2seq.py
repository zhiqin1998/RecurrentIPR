import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchtext.data.utils import get_tokenizer

from data.util import get_embedding_matrix, get_seq2seq_dataloader, get_word_coverage, get_seq2seq_trigger_dataloader
from models.layers.util import sign_to_str, calculate_ber
from models.nmt_seq2seq import Encoder, Decoder, KeyedEncoder, KeyedDecoder
from models.util import seed_everything, count_parameters, plot_weight_dist, replace_key
from trainer.nmt_seq2seq import EncDecTrainer, EncDecEvaluator, EncDecPrivateTrainer
from trainer.util import sequence_to_text
from key_config.util import get_seq2seq_keyed_kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, dest='train_file',
                        default='/datadrive/rnn-ipr/wmt14-enfr/train/combined_tokenized',
                        help='train file')
    parser.add_argument('--test-file', type=str, dest='test_file',
                        default='/datadrive/rnn-ipr/wmt14-enfr/test/newstest2014-tokenized',
                        help='test file')
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='output folder')
    parser.add_argument('--input-embedding-file', type=str, dest='input_embedding_file',
                        default='/datadrive/rnn-ipr/fasttext-embedding/cc.en.300.bin',
                        help='input pretrained fasttext embedding')
    parser.add_argument('--output-embedding-file', type=str, dest='output_embedding_file',
                        default='/datadrive/rnn-ipr/fasttext-embedding/cc.fr.300.bin',
                        help='output pretrained fasttext embedding')
    parser.add_argument('--src-vocab-path', type=str, dest='src_vocab_path', default='./outputs/enfr_en_vocab.pickle',
                        help='path to src vocab')
    parser.add_argument('--trg-vocab-path', type=str, dest='trg_vocab_path', default='./outputs/enfr_fr_vocab.pickle',
                        help='path to trg vocab')
    parser.add_argument('--src', type=str, dest='src', default='en', help='source language')
    parser.add_argument('--trg', type=str, dest='trg', default='fr', help='target language')
    parser.add_argument('--seed', type=int, dest='seed', default=1234, help='seed for experiment')
    parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='number of epochs')
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.3, help='dropout rate')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256, help='batch size per steps')
    parser.add_argument('--trigger-batch-size', type=int, dest='trigger_batch_size', default=0,
                        help='batch size per steps for trigger dataset')
    parser.add_argument('--latent-dim', type=int, dest='latent_dim', default=1024, help='hidden dimension of gru')
    parser.add_argument('--max-sentence-length', type=int, dest='max_sentence_length', default=15,
                        help='max sentence length')
    parser.add_argument('--bidirectional', action='store_true', dest='bidirectional', default=False,
                        help='use bidirectional encoder')
    parser.add_argument('--keyed-kwargs', type=str, dest='keyed_kwargs', default='', help='path to keyed_kwargs pickle file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    src_file = '{}.{}'.format(args.train_file, args.src)
    trg_file = '{}.{}'.format(args.train_file, args.trg)
    test_src_file = '{}.{}'.format(args.test_file, args.src)
    test_trg_file = '{}.{}'.format(args.test_file, args.trg)
    src_emb_file = args.input_embedding_file
    trg_emb_file = args.output_embedding_file
    max_vocab = 15000
    batch_size = args.batch_size
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    dropout = args.dropout
    hidden_dim = args.latent_dim
    embedding_dim = 300
    lr = 0.001
    grad_clip = 5.0
    teacher_forcing = 1.0
    bidirectional = args.bidirectional
    reverse_input = not bidirectional

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.isfile(args.keyed_kwargs):
        print('training baseline model')
    else:
        print('training private keyed model')
    if os.path.isfile(args.src_vocab_path) and os.path.isfile(args.trg_vocab_path):
        with open(args.src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(args.trg_vocab_path, 'rb') as f:
            trg_vocab = pickle.load(f)
        train_dataloader, valid_dataloader, _, _ = get_seq2seq_dataloader(src_file, trg_file, num_workers=2,
                                                                          in_vocab=src_vocab, out_vocab=trg_vocab,
                                                                          filters='•', num_sentences=1000000,
                                                                          in_tokenizer=get_tokenizer(None, 'en'),
                                                                          out_tokenizer=get_tokenizer(None, 'fr'),
                                                                          batch_size=batch_size,
                                                                          max_vocab=max_vocab,
                                                                          max_sentence_length=max_sentence_length,
                                                                          test_size=0.1,
                                                                          reverse_input=reverse_input
                                                                          )
    else:
        train_dataloader, valid_dataloader, src_vocab, trg_vocab = get_seq2seq_dataloader(src_file, trg_file,
                                                                                          num_workers=2,
                                                                                          filters='•',
                                                                                          in_tokenizer=get_tokenizer(
                                                                                              None,
                                                                                              'en'),
                                                                                          out_tokenizer=get_tokenizer(
                                                                                              None,
                                                                                              'fr'),
                                                                                          batch_size=batch_size,
                                                                                          max_vocab=max_vocab,
                                                                                          max_sentence_length=max_sentence_length,
                                                                                          test_size=0.1,
                                                                                          reverse_input=reverse_input
                                                                                          )
        with open(os.path.join(save_dir, 'enfr_en_vocab.pickle', 'wb')) as f:
            pickle.dump(src_vocab, f)
        with open(os.path.join(save_dir, 'enfr_fr_vocab.pickle', 'wb')) as f:
            pickle.dump(trg_vocab, f)

    num_words = min(max_vocab, len(src_vocab.itos))
    num_words_outputs = min(max_vocab, len(trg_vocab.itos))
    trg_pad_idx = trg_vocab.stoi['<pad>']
    trg_eos_idx = trg_vocab.stoi['<eos>']
    trg_sos_idx = trg_vocab.stoi['<sos>']

    print('first 50 src vocab:', src_vocab.itos[:50])
    print('first 50 trg vocab:', trg_vocab.itos[:50])

    print('source word coverage:', get_word_coverage(src_vocab, num_words))
    print('target word coverage:', get_word_coverage(trg_vocab, num_words_outputs))

    if not os.path.isfile(args.keyed_kwargs):
        encoder = Encoder(hidden_dim, embedding_dim,
                          get_embedding_matrix(src_emb_file, src_vocab, embedding_dim, max_vocab),
                          num_words, trg_pad_idx, bidirectional).to(device)
        decoder = Decoder(hidden_dim, embedding_dim,
                          get_embedding_matrix(trg_emb_file, trg_vocab, embedding_dim, max_vocab),
                          num_words_outputs, trg_pad_idx, dropout).to(device)
        print('encoder summary:')
        count_parameters(encoder)
        print('decoder summary:')
        count_parameters(decoder)

        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr)
        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr)
        enc_scheduler = StepLR(enc_optimizer, 7)
        dec_scheduler = StepLR(dec_optimizer, 7)
        criterion = nn.NLLLoss(ignore_index=trg_pad_idx)

        trainer = EncDecTrainer(encoder, decoder, enc_optimizer, dec_optimizer, criterion, device, grad_clip,
                                teacher_forcing=teacher_forcing)
        train_res, test_res = [], []
        for e in range(1, epochs + 1):
            tr = trainer.train(e, train_dataloader)
            te = trainer.test(valid_dataloader)
            train_res.append(tr)
            test_res.append(te)
            enc_scheduler.step()
            dec_scheduler.step()

        train_df = pd.DataFrame(train_res)
        test_df = pd.DataFrame(test_res)
        train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
        test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))
        
        print('average training time per epoch:', train_df['time'].mean())

        torch.save(encoder, os.path.join(save_dir, 'encoder_{}.pth'.format(seed)))
        torch.save(decoder, os.path.join(save_dir, 'decoder_{}.pth'.format(seed)))

        evaluator = EncDecEvaluator(encoder, decoder, device, trg_vocab)
        test_dataloader, _, _ = get_seq2seq_dataloader(test_src_file, test_trg_file, in_vocab=src_vocab,
                                                       out_vocab=trg_vocab,
                                                       filters='•',
                                                       in_tokenizer=get_tokenizer(None, 'en'),
                                                       out_tokenizer=get_tokenizer(None, 'fr'),
                                                       batch_size=batch_size, max_vocab=max_vocab, shuffle=False,
                                                       max_sentence_length=max_sentence_length, test_size=None,
                                                       reverse_input=reverse_input
                                                       )

        res = evaluator.evaluate_bleu(test_dataloader)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(encoder.gru, ax, bins=100)
        plt.title('Encoder Weight')
        plt.savefig(os.path.join(save_dir, 'encoder_weight.png'))
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(decoder.gru, ax, bins=100)
        plt.title('Decoder Weight')
        plt.savefig(os.path.join(save_dir, 'decoder_weight.png'))
    else:
        keyed_kwargs = get_seq2seq_keyed_kwargs(args.keyed_kwargs, src_vocab, trg_vocab)
        print('encoder key:')
        print(sequence_to_text(keyed_kwargs['enc_key'], src_vocab))
        print('\ndecoder key:')
        print(sequence_to_text(keyed_kwargs['dec_key'], trg_vocab))
        with open(os.path.join(save_dir, 'keyed_kwargs_{}.pickle'.format(seed)), 'wb') as f:
            pickle.dump(keyed_kwargs, f)

        if args.trigger_batch_size > 0:
            # create trigger dataloader randomly
            trigger_dataloader = get_seq2seq_trigger_dataloader(n=100, in_vocab=src_vocab, out_vocab=trg_vocab,
                                                        batch_size=args.trigger_batch_size, reverse_input=reverse_input)
            torch.save(trigger_dataloader, os.path.join(save_dir, 'trigger_dataloader_{}.pth'.format(seed)))
            print('training with trigger dataset')
        else:
            trigger_dataloader = None

        encoder = KeyedEncoder(hidden_dim, embedding_dim, get_embedding_matrix(src_emb_file, src_vocab, embedding_dim, max_vocab),
                               num_words, trg_pad_idx, bidirectional, keyed_kwargs).to(device)
        decoder = KeyedDecoder(hidden_dim, embedding_dim, get_embedding_matrix(trg_emb_file, trg_vocab, embedding_dim, max_vocab),
                               num_words_outputs, trg_pad_idx, dropout, keyed_kwargs).to(device)
        print('encoder summary:')
        count_parameters(encoder)
        print('decoder summary:')
        count_parameters(decoder)

        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr)
        dec_optimizer = torch.optim.Adam(decoder.parameters(), lr)
        enc_scheduler = StepLR(enc_optimizer, 7)
        dec_scheduler = StepLR(dec_optimizer, 7)
        criterion = nn.NLLLoss(ignore_index=trg_pad_idx)
        trainer = EncDecPrivateTrainer(encoder, decoder, enc_optimizer, dec_optimizer, criterion, device, grad_clip,
                                       teacher_forcing=teacher_forcing)

        if trigger_dataloader:
            train_res, test_res, trigger_res = [], [], []
            for e in range(1, epochs + 1):
                tra = trainer.train(e, train_dataloader, trigger_dataloader, reverse_input, trg_pad_idx)
                tes = trainer.test(valid_dataloader)
                tri = trainer.test(trigger_dataloader, msg='Trigger testing')
                train_res.append(tra)
                test_res.append(tes)
                trigger_res.append(tri)
                enc_scheduler.step()
                dec_scheduler.step()

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
                enc_scheduler.step()
                dec_scheduler.step()
            train_df = pd.DataFrame(train_res)
            test_df = pd.DataFrame(test_res)
            train_df.to_csv(os.path.join(save_dir, 'train_{}.csv'.format(seed)))
            test_df.to_csv(os.path.join(save_dir, 'valid_{}.csv'.format(seed)))
            
        print('average training time per epoch:', train_df['time'].mean())

        torch.save(encoder, os.path.join(save_dir, 'encoder_{}.pth'.format(seed)))
        torch.save(decoder, os.path.join(save_dir, 'decoder_{}.pth'.format(seed)))

        evaluator = EncDecEvaluator(encoder, decoder, device, trg_vocab)
        test_dataloader, _, _ = get_seq2seq_dataloader(test_src_file, test_trg_file, in_vocab=src_vocab,
                                                       out_vocab=trg_vocab,
                                                       filters='•',
                                                       in_tokenizer=get_tokenizer(None, 'en'),
                                                       out_tokenizer=get_tokenizer(None, 'fr'),
                                                       batch_size=batch_size, max_vocab=max_vocab, shuffle=False,
                                                       max_sentence_length=max_sentence_length, test_size=None,
                                                       reverse_input=reverse_input
                                                       )

        print('Public test evaluation')
        res = evaluator.evaluate_bleu(test_dataloader, use_key=False)
        print('Private test evaluation')
        res = evaluator.evaluate_bleu(test_dataloader, use_key=True)
        if trigger_dataloader:
            print('Public trigger evaluation')
            res = evaluator.evaluate_bleu(trigger_dataloader, use_key=False)
            print('Private trigger evaluation')
            res = evaluator.evaluate_bleu(trigger_dataloader, use_key=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(encoder.gru, ax, bins=100)
        plt.title('Encoder Weight')
        plt.savefig(os.path.join(save_dir, 'encoder_weight.png'))
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_weight_dist(decoder.gru, ax, bins=100)
        plt.title('Decoder Weight')
        plt.savefig(os.path.join(save_dir, 'decoder_weight.png'))

        try:
            print(sign_to_str(encoder.get_signature().cpu().detach().numpy(), len(keyed_kwargs['signature'])))
        except UnicodeError:
            pass
        print('encoder signature ber:',
              calculate_ber(encoder.get_signature().cpu().detach().numpy(), keyed_kwargs['signature']))
        try:
            print(sign_to_str(decoder.get_signature().cpu().detach().numpy(), len(keyed_kwargs['signature'])))
        except UnicodeError:
            pass
        print('decoder signature ber:',
              calculate_ber(decoder.get_signature().cpu().detach().numpy(), keyed_kwargs['signature']))

        print('random key attack')
        # random key
        encoder.set_key(torch.randint(max(0, num_words - 5000), num_words, (8, 15)).to(device))
        decoder.set_key(torch.randint(max(0, num_words - 5000), num_words_outputs, (8, 20)).to(device))
        print('Private test evaluation')
        res = evaluator.evaluate_bleu(test_dataloader, use_key=True)
        if trigger_dataloader:
            print('Private trigger evaluation')
            res = evaluator.evaluate_bleu(trigger_dataloader, use_key=True)

        print('70% correct key attack')
        # random key
        thres = 1 - 0.7
        encoder.set_key(replace_key(keyed_kwargs['enc_key'], num_words, perc=thres).to(device))
        decoder.set_key(replace_key(keyed_kwargs['dec_key'], num_words_outputs, perc=thres).to(device))
        print('Private test evaluation')
        res = evaluator.evaluate_bleu(test_dataloader, use_key=True)
        if trigger_dataloader:
            print('Private trigger evaluation')
            res = evaluator.evaluate_bleu(trigger_dataloader, use_key=True)
