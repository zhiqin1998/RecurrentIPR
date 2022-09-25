import argparse
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

from data.util import get_seq2seq_dataloader
from models.layers.util import sign_to_str, calculate_ber
from models.util import seed_everything
from trainer.nmt_seq2seq import EncDecTrainer, EncDecEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, dest='train_file',
                        default='/datadrive/rnn-ipr/wmt14-enfr/train/finetune_tokenized',
                        help='train file')
    parser.add_argument('--test-file', type=str, dest='test_file',
                        default='/datadrive/rnn-ipr/wmt14-enfr/test/newstest2014-tokenized',
                        help='test file')
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='output folder')
    parser.add_argument('--src', type=str, dest='src', default='en', help='source language')
    parser.add_argument('--trg', type=str, dest='trg', default='fr', help='target language')
    parser.add_argument('--src-vocab-path', type=str, dest='src_vocab_path', default='./outputs/enfr_en_vocab.pickle',
                        help='path to src vocab')
    parser.add_argument('--trg-vocab-path', type=str, dest='trg_vocab_path', default='./outputs/enfr_fr_vocab.pickle',
                        help='path to trg vocab')
    parser.add_argument('--seed', type=int, dest='seed', default=1234, help='seed for experiment')
    parser.add_argument('--epochs', type=int, dest='epochs', default=5, help='number of epochs')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256, help='batch size per steps')
    parser.add_argument('--max-sentence-length', type=int, dest='max_sentence_length', default=15,
                        help='max sentence length')
    parser.add_argument('--num-training-sentences', type=int, dest='num_sentences', default=1500000,
                        help='number of training example to use')
    parser.add_argument('--reverse-input', action='store_true', dest='reverse_input', default=True,
                        help='reverse input sequence')
    parser.add_argument('--pretrained-path', type=str, dest='pretrained_path', help='path to saved pretrained model',
                        required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    src_file = '{}.{}'.format(args.train_file, args.src)
    trg_file = '{}.{}'.format(args.train_file, args.trg)
    test_src_file = '{}.{}'.format(args.test_file, args.src)
    test_trg_file = '{}.{}'.format(args.test_file, args.trg)
    batch_size = args.batch_size
    epochs = args.epochs
    max_sentence_length = args.max_sentence_length
    lr = 0.00001
    max_vocab = 15000
    grad_clip = 5.0
    teacher_forcing = 1.0
    reverse_input = args.reverse_input

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(args.src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.trg_vocab_path, 'rb') as f:
        trg_vocab = pickle.load(f)

    train_dataloader, valid_dataloader, _, _ = get_seq2seq_dataloader(src_file, trg_file, num_workers=2,
                                                                      in_vocab=src_vocab, out_vocab=trg_vocab,
                                                                      filters='•', num_sentences=args.num_sentences,
                                                                      in_tokenizer=get_tokenizer(None, 'en'),
                                                                      out_tokenizer=get_tokenizer(None, 'fr'),
                                                                      batch_size=batch_size, max_vocab=max_vocab,
                                                                      max_sentence_length=max_sentence_length,
                                                                      test_size=0.1,
                                                                      reverse_input=reverse_input
                                                                      )
    if os.path.isfile(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed))):
        print('found trigger dataset')
        trigger_dataloader = torch.load(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed)))
    else:
        trigger_dataloader = None

    num_words = min(max_vocab, len(src_vocab.itos))
    num_words_outputs = min(max_vocab, len(trg_vocab.itos))
    trg_pad_idx = trg_vocab.stoi['<pad>']
    trg_eos_idx = trg_vocab.stoi['<eos>']
    trg_sos_idx = trg_vocab.stoi['<sos>']

    with open(os.path.join(args.pretrained_path, 'keyed_kwargs_{}.pickle'.format(seed)), 'rb') as f:
        keyed_kwargs = pickle.load(f)

    encoder = torch.load(os.path.join(args.pretrained_path, 'encoder_{}.pth'.format(seed)))
    decoder = torch.load(os.path.join(args.pretrained_path, 'decoder_{}.pth'.format(seed)))
    # remove sign loss from models
    encoder.gru.sign_loss = None
    decoder.gru.sign_loss = None

    keyed_kwargs['enc_key'] = encoder.key.cpu().clone()
    keyed_kwargs['dec_key'] = decoder.key.cpu().clone()

    # finetune with small learning rate
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr)
    criterion = nn.NLLLoss(ignore_index=trg_pad_idx)
    trainer = EncDecTrainer(encoder, decoder, enc_optimizer, dec_optimizer, criterion, device, grad_clip,
                            teacher_forcing=teacher_forcing)
    train_res, test_res, trigger_res = [], [], []
    for e in range(1, epochs + 1):
        tra = trainer.train(e, train_dataloader, use_key=False)
        tes = trainer.test(valid_dataloader, use_key=False)
        if trigger_dataloader:
            tri = trainer.test(trigger_dataloader, use_key=False, msg='Trigger testing')
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
