import argparse
import os
import pickle

import pandas as pd
import torch
import torch.nn.utils.prune as prune
from torchtext.data.utils import get_tokenizer

from data.util import get_seq2seq_dataloader
from models.losses.sign_loss import SignLoss
from models.util import seed_everything
from trainer.nmt_seq2seq import EncDecEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256, help='batch size per steps')
    parser.add_argument('--max-sentence-length', type=int, dest='max_sentence_length', default=15,
                        help='max sentence length')
    parser.add_argument('--reverse-input', action='store_true', dest='reverse_input', default=True,
                        help='reverse input sequence')
    parser.add_argument('--pretrained-path', type=str, dest='pretrained_path', help='path to saved pretrained model',
                        required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    batch_size = args.batch_size
    max_vocab = 15000
    test_src_file = '{}.{}'.format(args.test_file, args.src)
    test_trg_file = '{}.{}'.format(args.test_file, args.trg)
    max_sentence_length = args.max_sentence_length
    reverse_input = args.reverse_input

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(args.src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.trg_vocab_path, 'rb') as f:
        trg_vocab = pickle.load(f)

    num_words = min(max_vocab, len(src_vocab.itos))
    num_words_outputs = min(max_vocab, len(trg_vocab.itos))
    trg_pad_idx = trg_vocab.stoi['<pad>']
    trg_eos_idx = trg_vocab.stoi['<eos>']
    trg_sos_idx = trg_vocab.stoi['<sos>']

    with open(os.path.join(args.pretrained_path, 'keyed_kwargs_{}.pickle'.format(seed)), 'rb') as f:
        keyed_kwargs = pickle.load(f)
    # loading dataset
    test_dataloader, _, _ = get_seq2seq_dataloader(test_src_file, test_trg_file, in_vocab=src_vocab,
                                                   out_vocab=trg_vocab,
                                                   filters='•',
                                                   in_tokenizer=get_tokenizer(None, 'en'),
                                                   out_tokenizer=get_tokenizer(None, 'fr'),
                                                   batch_size=batch_size, max_vocab=max_vocab, shuffle=False,
                                                   max_sentence_length=max_sentence_length, test_size=None,
                                                   reverse_input=reverse_input
                                                   )

    if os.path.isfile(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed))):
        print('found trigger dataset')
        trigger_dataloader = torch.load(os.path.join(args.pretrained_path, 'trigger_dataloader_{}.pth'.format(seed)))
    else:
        trigger_dataloader = None

    res = []

    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        encoder = torch.load(os.path.join(args.pretrained_path, 'encoder_{}.pth'.format(seed)))
        decoder = torch.load(os.path.join(args.pretrained_path, 'decoder_{}.pth'.format(seed)))

        # pruning
        param_to_prune = [(encoder.input_embedding, 'weight'), (decoder.output_embedding, 'weight'),
                          (decoder.classifier, 'weight')]
        for param, _ in encoder.gru.cell.named_parameters():
            param_to_prune.append((encoder.gru.cell, param))
        for param, _ in decoder.gru.cell.named_parameters():
            param_to_prune.append((decoder.gru.cell, param))
        prune.global_unstructured(
            param_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=perc,
        )
        print('*' * 50)
        print(f'Evaluating with {perc} pruned weight:')
        evaluator = EncDecEvaluator(encoder, decoder, device, trg_vocab)
        te = evaluator.evaluate_bleu(test_dataloader, use_key=True)
        enc_sign_acc = encoder.gru.sign_loss.acc.item()
        dec_sign_acc = decoder.gru.sign_loss.acc.item()
        if trigger_dataloader is not None:
            tri = evaluator.evaluate_bleu(trigger_dataloader, use_key=True)
            res.append({'prune_perc': perc, 'enc_sign_acc': enc_sign_acc, 'dec_sign_acc': dec_sign_acc,
                        'test_bleu': te['bleu_score'], 'trigger_bleu': tri['bleu_score']})
        else:
            res.append({'prune_perc': perc, 'enc_sign_acc': enc_sign_acc, 'dec_sign_acc': dec_sign_acc,
                        'test_bleu': te['bleu_score']})

    train_df = pd.DataFrame(res)
    train_df.to_csv(os.path.join(save_dir, 'results_{}.csv'.format(seed)))
