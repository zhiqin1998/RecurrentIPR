import torch
import random
import fasttext
import numpy as np
import pandas as pd

from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from sklearn.model_selection import train_test_split


def get_embedding_matrix(vec_filepath, vocab, embedding_dim=300, max_vocab_size=15000, mean=0.0, std=0.05,
                         specials=('<eos>', '<sos>')):
    # function to load pretrained fasttext word embedding
    try:
        vec = fasttext.load_model(vec_filepath)
        if embedding_dim != 300:
            fasttext.util.reduce_model(vec, embedding_dim)
        num_words = min(max_vocab_size, len(vocab.stoi) + 1)
        embedding_matrix = np.zeros((num_words, embedding_dim))
        unk_vector = []
        for word, _ in vocab.freqs.most_common(num_words * 2):
            embedding_vector = vec.get_word_vector(word)
            index = vocab.stoi[word]
            if index == vocab.stoi['<unk>']:
                unk_vector.append(embedding_vector)
            elif embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                embedding_matrix[index] = np.random.normal(mean, std, embedding_dim)
                print(word)
        # create embedding for unknown token by averaging all oov word embedding
        if len(unk_vector):
            embedding_matrix[vocab.stoi['<unk>']] = np.stack(unk_vector, axis=0).mean(axis=0)
        else:
            embedding_matrix[vocab.stoi['<unk>']] = np.random.normal(mean, std, embedding_dim)

        if specials is not None:
            # set random value for special tokens
            for s in specials:
                embedding_matrix[vocab.stoi[s]] = np.random.normal(mean, std, embedding_dim)
                embedding_matrix[vocab.stoi[s]] = np.random.normal(mean, std, embedding_dim)

        return embedding_matrix
    except:
        print('error loading pretrained embeddeding, using random weights')
        return None


def get_word_coverage(vocab, max_vocab_size):
    # calculating word coverage give the maximum vocab size
    total_wc = sum(vocab.freqs.values())
    covered = sum([i for _, i in vocab.freqs.most_common(max_vocab_size)])
    return covered / total_wc * 100


def read_parallel_corpus(input_data_path, output_data_path, filters='', num_sentences=float('inf'),
                         max_sentence_length=15):
    # function to read a read and process a pair of parallel corpus
    input_sentences = []
    output_sentences = []
    translate_dict = {c: '' for c in filters}
    translate_map = str.maketrans(translate_dict)
    i = 0
    with open(input_data_path, encoding="utf-8", newline='\n') as src_file, open(output_data_path, encoding="utf-8",
                                                                                 newline='\n') as trg_file:
        for input_sentence, output in zip(src_file, trg_file):
            input_sentence, output = input_sentence.translate(translate_map).strip().lower(), output.translate(
                translate_map).strip().lower()

            if input_sentence == '' or output == '':
                continue
            # skip training pair that has long length
            if len(input_sentence.split()) > max_sentence_length or len(output.split()) > 1.5 * max_sentence_length:
                continue

            output_sentence = output

            input_sentences.append(input_sentence)
            output_sentences.append(output_sentence)
            i += 1
            if i >= num_sentences:
                break
    return input_sentences, output_sentences


def build_vocab(sentences, tokenizer, max_size=None, specials=None):
    # build pytorch vocab object
    counter = Counter()
    for string_ in sentences:
        counter.update(tokenizer(string_))
    if max_size is not None and specials is not None:
        max_size = max_size - len(specials)
    return Vocab(counter, max_size, specials=specials)


def data_process_seq2seq(input_sentences, output_sentences, in_vocab, out_vocab, in_tokenizer, out_tokenizer):
    # function to covert word token to integer with vocabulary
    data = []
    for in_s, out_s in zip(input_sentences, output_sentences):
        in_tensor_ = torch.tensor([in_vocab[token] for token in in_tokenizer(in_s)],
                                  dtype=torch.long)
        out_tensor_ = torch.tensor([out_vocab[token] for token in out_tokenizer(out_s)],
                                   dtype=torch.long)
        data.append((in_tensor_, out_tensor_))
    return data


class Seq2seqBatch:
    # callable class for collate fn so that it is pickleable
    def __init__(self, sos_idx, eos_idx, pad_idx, reverse_input):
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.reverse_input = reverse_input

    def __call__(self, data_batch):
        in_data, out_data, out_in_data = [], [], []
        for in_d, out_d in data_batch:
            in_data.append(torch.cat([in_d, torch.tensor([self.eos_idx])], dim=-1))
            out_data.append(torch.cat([out_d, torch.tensor([self.eos_idx])], dim=-1))
            out_in_data.append(torch.cat([torch.tensor([self.sos_idx]), out_d], dim=-1))
        if self.reverse_input:
            in_data = torch.flip(pad_sequence(in_data, batch_first=True, padding_value=self.pad_idx), (-1,))
        else:
            in_data = pad_sequence(in_data, batch_first=True, padding_value=self.pad_idx)
        out_in_data = pad_sequence(out_in_data, batch_first=True, padding_value=self.pad_idx)
        out_data = pad_sequence(out_data, batch_first=True, padding_value=self.pad_idx)
        return in_data, out_in_data, out_data


def get_seq2seq_trigger_dataloader(n=100, sentence_length_range=(15, 20), in_vocab=None, out_vocab=None, batch_size=2,
                                   reverse_input=True, shuffle=False):
    # function to create trigger dataset randomly
    data = []
    for _ in range(n):
        in_l = random.randint(sentence_length_range[0], sentence_length_range[1])
        in_tensor_ = torch.randint(4, len(in_vocab.itos), (in_l,), dtype=torch.long)
        out_l = random.randint(sentence_length_range[0], sentence_length_range[1])
        out_tensor_ = torch.randint(4, len(out_vocab.itos), (out_l,), dtype=torch.long)
        data.append((in_tensor_, out_tensor_))
    sos_idx = out_vocab['<sos>']
    eos_idx = out_vocab['<eos>']
    pad_idx = out_vocab['<pad>']
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=Seq2seqBatch(sos_idx, eos_idx, pad_idx, reverse_input))


def get_seq2seq_dataloader(input_data_path, output_data_path, filters='', num_sentences=float('inf'), test_size=0.2,
                           num_workers=0,
                           max_sentence_length=15, max_vocab=None, in_tokenizer=None, out_tokenizer=None, batch_size=64,
                           shuffle=True, in_vocab=None, out_vocab=None, reverse_input=True):
    # function to load training and validation dataloader for the parallel corpus
    if in_tokenizer is None:
        in_tokenizer = get_tokenizer(None, 'en')
    if out_tokenizer is None:
        out_tokenizer = get_tokenizer(None, 'fr')
    input_sentences, output_sentences = read_parallel_corpus(input_data_path, output_data_path, filters, num_sentences,
                                                             max_sentence_length)
    if in_vocab is None:
        in_vocab = build_vocab(input_sentences, in_tokenizer, max_vocab, specials=['<unk>', '<pad>', '<eos>'])
    if out_vocab is None:
        out_vocab = build_vocab(output_sentences, out_tokenizer, max_vocab,
                                specials=['<unk>', '<pad>', '<eos>', '<sos>'])

    sos_idx = out_vocab['<sos>']
    eos_idx = out_vocab['<eos>']
    pad_idx = out_vocab['<pad>']

    if test_size is None:
        return DataLoader(
            data_process_seq2seq(input_sentences, output_sentences, in_vocab, out_vocab, in_tokenizer, out_tokenizer),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=Seq2seqBatch(sos_idx, eos_idx, pad_idx, reverse_input)), in_vocab, out_vocab

    else:
        train_input_sentences, test_input_sentences, train_output_sentences, test_output_sentences = train_test_split(
            input_sentences, output_sentences, test_size=test_size)

        return DataLoader(
            data_process_seq2seq(train_input_sentences, train_output_sentences, in_vocab, out_vocab, in_tokenizer,
                                 out_tokenizer),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=Seq2seqBatch(sos_idx, eos_idx, pad_idx, reverse_input)), DataLoader(
            data_process_seq2seq(test_input_sentences, test_output_sentences, in_vocab, out_vocab, in_tokenizer,
                                 out_tokenizer),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=Seq2seqBatch(sos_idx, eos_idx, pad_idx, reverse_input)), in_vocab, out_vocab


class Batch:
    # callable class for collate fn so that it is pickleable
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, data_batch):
        sentences, labels = [], []
        for sent, label in data_batch:
            sentences.append(sent)
            labels.append(label)
        sentences = pad_sequence(sentences, batch_first=True, padding_value=self.pad_idx)
        labels = torch.LongTensor(labels)
        return sentences, labels


def get_trec_dataset(num_workers=0, batch_size=64, max_sentence_length=30, max_vocab=None, trigger_size=None,
                     trigger_batch_size=1):
    # load trec-6 dataset from online source
    train_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
    test_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
    train_df = pd.read_csv(train_url, sep=':', header=None, names=['labels', 'sentences'], encoding='ISO-8859-1')
    test_df = pd.read_csv(test_url, sep=':', header=None, names=['labels', 'sentences'], encoding='ISO-8859-1')
    train_df['sentences'] = train_df['sentences'].apply(lambda x: ' '.join(x.lower().split()[1:max_sentence_length]))
    test_df['sentences'] = test_df['sentences'].apply(lambda x: ' '.join(x.lower().split()[1:max_sentence_length]))
    class_label = {'ABBR': 0, 'DESC': 1, 'ENTY': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
    train_df['labels'] = train_df['labels'].map(class_label)
    test_df['labels'] = test_df['labels'].map(class_label)

    tokenizer = get_tokenizer(None, 'en')
    vocab = build_vocab(train_df['sentences'].values, tokenizer, max_size=max_vocab, specials=('<unk>', '<pad>'))
    pad_idx = vocab['<pad>']
    if trigger_size is not None:
        # create trigger dataset by randomly sampling from training dataset and changing their labels
        train_df, trigger_df = train_test_split(train_df, test_size=trigger_size, stratify=train_df['labels'])
        trigger_label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 0}
        trigger_df['labels'] = trigger_df['labels'].map(trigger_label)

        return DataLoader(data_process_sa(train_df['sentences'].values, train_df['labels'].values, vocab, tokenizer),
                          batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          collate_fn=Batch(pad_idx)), DataLoader(
            data_process_sa(test_df['sentences'].values, test_df['labels'].values, vocab, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Batch(pad_idx)), DataLoader(
            data_process_sa(trigger_df['sentences'].values, trigger_df['labels'].values, vocab, tokenizer),
            batch_size=trigger_batch_size,
            shuffle=False,
            collate_fn=Batch(pad_idx)), vocab
    else:
        return DataLoader(data_process_sa(train_df['sentences'].values, train_df['labels'].values, vocab, tokenizer),
                          batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          collate_fn=Batch(pad_idx)), DataLoader(
            data_process_sa(test_df['sentences'].values, test_df['labels'].values, vocab, tokenizer),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Batch(pad_idx)), vocab


def data_process_sa(sentences, labels, vocab, tokenizer):
    # function to process work tokens to integer
    data = []
    for sent, label in zip(sentences, labels):
        sent_tensor_ = torch.tensor([vocab[token] for token in tokenizer(sent)],
                                    dtype=torch.long)
        data.append((sent_tensor_, label))
    return data


class Transform:
    def __init__(self, pixel=False):
        self.pixel = pixel

    def __call__(self, x):
        if self.pixel:
            return x.view(-1, 1)
        else:
            return x.squeeze()


class MNISTTriggerCol:
    def __init__(self):
        self.noise = torch.normal(0, 0.05, (28, 28))

    def __call__(self, batch):
        imgs, labels = [], []
        for img, label in batch:
            imgs.append(img + self.noise)
            labels.append(label)
        labels = torch.LongTensor(labels)
        return torch.stack(imgs), labels


def get_mnist_dataset(num_workers=0, batch_size=64, trigger_size=None, trigger_batch_size=1, pixel=False):
    transform = [ToTensor(), Transform(pixel)]
    train_dataset = MNIST('.data/', train=True, download=True, transform=Compose(transform))
    test_dataset = MNIST('.data/', train=False, download=True, transform=Compose(transform))
    if trigger_size is not None:
        train_idx, trigger_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=trigger_size,
            shuffle=True,
            stratify=train_dataset.targets)
        train_dataset.targets[trigger_idx] += 1
        train_dataset.targets[train_dataset.targets == 10] = 0
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        trigger_sampler = torch.utils.data.SubsetRandomSampler(trigger_idx)

        return DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                          sampler=train_sampler), DataLoader(test_dataset, batch_size=batch_size,
                                                             shuffle=False), DataLoader(train_dataset,
                                                                                        collate_fn=MNISTTriggerCol(),
                                                                                        batch_size=trigger_batch_size,
                                                                                        sampler=trigger_sampler)
    else:
        return DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True), DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
