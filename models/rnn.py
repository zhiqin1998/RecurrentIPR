import torch
import torch.nn as nn

from models.layers.normal import LSTM, GRU
from models.layers.keyed import KeyedLSTM, KeyedGRU


class BiRNN(nn.Module):
    def __init__(self, hidden_dim=16, embedding_dim=300, output_dim=2, input_embedding_matrix=None, num_words=15000,
                 pad_idx=1, dropout=0.4, embedding_dropout=0.5, rnn_type='lstm'):
        super(BiRNN, self).__init__()
        if input_embedding_matrix is not None:
            self.input_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(input_embedding_matrix).to(torch.float32), padding_idx=pad_idx, freeze=False)
        else:
            self.input_embedding = nn.Embedding(num_words, embedding_dim, padding_idx=pad_idx)
        if rnn_type == 'lstm':
            self.rnn = LSTM(embedding_dim, hidden_dim)
        else:
            self.rnn = GRU(embedding_dim, hidden_dim)
        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        embedded = self.emb_dropout(self.input_embedding(x))
        outputs = self.rnn(embedded)
        reversed_embedded = torch.flip(embedded, (1,))
        reversed_outputs = self.rnn(reversed_embedded)
        # concatenate final output from both direction
        output = torch.cat((outputs, reversed_outputs), 1)
        output = self.classifier(self.dropout(output))
        return output


class KeyedBiRNN(BiRNN):
    def __init__(self, hidden_dim=16, embedding_dim=300, output_dim=2, input_embedding_matrix=None, num_words=15000,
                 pad_idx=1, dropout=0.4, embedding_dropout=0.5, keyed_kwargs=None, rnn_type='lstm'):
        super(KeyedBiRNN, self).__init__(hidden_dim, embedding_dim, output_dim, input_embedding_matrix, num_words,
                                         pad_idx, dropout, embedding_dropout, rnn_type)
        if rnn_type == 'lstm':
            self.rnn = KeyedLSTM(embedding_dim, hidden_dim, keyed_kwargs=keyed_kwargs)
        else:
            self.rnn = KeyedGRU(embedding_dim, hidden_dim, keyed_kwargs=keyed_kwargs)
        self.register_buffer('key', keyed_kwargs.get('key', torch.randint(0, num_words, (4, 150))))

    def set_key(self, key):
        self.register_buffer('key', key)

    def get_signature(self, reduce=True):
        if self.rnn_type == 'lstm':
            if reduce:
                return self.rnn.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1][0].mean(dim=0)
            else:
                return self.rnn.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1][0]
        else:
            if reduce:
                return self.rnn.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1].mean(dim=0)
            else:
                return self.rnn.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1]

    def forward(self, x, use_key=True):
        embedded = self.emb_dropout(self.input_embedding(x))
        reversed_embedded = torch.flip(embedded, (1,))
        if use_key:
            key = self.input_embedding(self.key)
            reversed_outputs = self.rnn(reversed_embedded, key=key)
            outputs = self.rnn(embedded, key=key)
        else:
            reversed_outputs = self.rnn(reversed_embedded)
            outputs = self.rnn(embedded)
        # concatenate final output from both direction
        output = torch.cat((outputs, reversed_outputs), 1)
        output = self.classifier(self.dropout(output))
        return output


class MNISTRNN(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=28, output_dim=10, dropout=0.4, rnn_type='lstm'):
        super(MNISTRNN, self).__init__()
        if rnn_type == 'lstm':
            self.rnn = LSTM(input_dim, hidden_dim)
        else:
            self.rnn = GRU(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        outputs = self.rnn(x)
        output = self.classifier(self.dropout(outputs))
        return output


class MNISTKeyedRNN(MNISTRNN):
    def __init__(self, hidden_dim=128, input_dim=28, output_dim=10, dropout=0.4, rnn_type='lstm', keyed_kwargs=None):
        super(MNISTKeyedRNN, self).__init__(hidden_dim, input_dim, output_dim, dropout, rnn_type)
        if rnn_type == 'lstm':
            self.rnn = KeyedLSTM(input_dim, hidden_dim, keyed_kwargs=keyed_kwargs)
        else:
            self.rnn = KeyedGRU(input_dim, hidden_dim, keyed_kwargs=keyed_kwargs)
        self.register_buffer('key', keyed_kwargs.get('key', torch.randn((8, 28, 28))))

    def set_key(self, key):
        self.register_buffer('key', key)

    def get_signature(self, reduce=True):
        if self.rnn_type == 'lstm':
            if reduce:
                return self.rnn.get_key_gate(self.key, return_first_hidden=True)[1][0].mean(dim=0)
            else:
                return self.rnn.get_key_gate(self.key, return_first_hidden=True)[1][0]
        else:
            if reduce:
                return self.rnn.get_key_gate(self.key, return_first_hidden=True)[1].mean(dim=0)
            else:
                return self.rnn.get_key_gate(self.key, return_first_hidden=True)[1]

    def forward(self, x, use_key=True):
        if use_key:
            outputs = self.rnn(x, key=self.key)
        else:
            outputs = self.rnn(x)
        output = self.classifier(self.dropout(outputs))
        return output
