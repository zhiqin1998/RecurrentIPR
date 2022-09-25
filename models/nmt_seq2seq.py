import torch
import torch.nn as nn

from models.layers.normal import GRU
from models.layers.keyed import KeyedGRU


class Encoder(nn.Module):
    def __init__(self, hidden_dim=1024, embedding_dim=300, input_embedding_matrix=None, num_words=15000, pad_idx=1,
                 bidirectional=False):
        super(Encoder, self).__init__()
        # load from pretrained embedding if available
        if input_embedding_matrix is not None:
            self.input_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(input_embedding_matrix).to(torch.float32), padding_idx=pad_idx, freeze=False)
        else:
            self.input_embedding = nn.Embedding(num_words, embedding_dim, padding_idx=pad_idx)
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0
            self.gru = GRU(embedding_dim, hidden_dim / 2)
        else:
            self.gru = GRU(embedding_dim, hidden_dim)

    def forward(self, x):
        embedded = self.input_embedding(x)
        output = self.gru(embedded)
        if self.bidirectional:
            reversed_x = torch.flip(x, (-1,))
            reversed_embedded = self.input_embedding(reversed_x)
            reversed_output = self.gru(reversed_embedded)
            output = torch.cat((output, reversed_output), 1)
        return output


class KeyedEncoder(Encoder):
    def __init__(self, hidden_dim=1024, embedding_dim=300, input_embedding_matrix=None, num_words=15000, pad_idx=1,
                 bidirectional=False, keyed_kwargs=None):
        super(KeyedEncoder, self).__init__(hidden_dim, embedding_dim, input_embedding_matrix, num_words, pad_idx,
                                           bidirectional)
        if self.bidirectional:
            self.gru = KeyedGRU(embedding_dim, hidden_dim / 2, keyed_kwargs)
        else:
            self.gru = KeyedGRU(embedding_dim, hidden_dim, keyed_kwargs)
        self.register_buffer('key', keyed_kwargs.get('enc_key', torch.randint(0, num_words, (1, 15))))

    def set_key(self, key):
        self.register_buffer('key', key)

    def get_signature(self, reduce=True):
        if reduce:
            return self.gru.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1].mean(dim=0)
        else:
            return self.gru.get_key_gate(self.input_embedding(self.key), return_first_hidden=True)[1]

    def forward(self, x, use_key=True):
        embedded = self.input_embedding(x)
        if use_key:
            key = self.input_embedding(self.key)
            output = self.gru(embedded, key=key)
        else:
            output = self.gru(embedded, )
        if self.bidirectional:
            reversed_x = torch.flip(x, (-1,))
            reversed_embedded = self.gru(reversed_x)
            if use_key:
                reversed_key = torch.flip(self.key, (-1,))
                key = self.input_embedding(reversed_key)
                reversed_output = self.gru(reversed_embedded, key=key)
            else:
                reversed_output = self.gru(reversed_embedded)
            output = torch.cat((output, reversed_output), 1)
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_dim=1024, embedding_dim=300, output_embedding_matrix=None, num_words_outputs=15000,
                 pad_idx=1, dropout=0.3):
        super(Decoder, self).__init__()
        # load from pretrained embedding if available
        if output_embedding_matrix is not None:
            self.output_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(output_embedding_matrix).to(torch.float32),
                padding_idx=pad_idx, freeze=False)
        else:
            self.output_embedding = nn.Embedding(num_words_outputs, embedding_dim, padding_idx=pad_idx)
        self.gru = GRU(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_words_outputs)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # decoder forward is implemented in timestep fashion
        embedded = self.output_embedding(x)
        dec_hidden = self.gru(embedded, init_hidden=hidden, teacher_forcing=False)
        output = self.softmax(self.classifier(self.dropout(dec_hidden)))
        return output, dec_hidden


class KeyedDecoder(Decoder):
    def __init__(self, hidden_dim=1024, embedding_dim=300, output_embedding_matrix=None, num_words_outputs=15000,
                 pad_idx=1, dropout=0.3, keyed_kwargs=None):
        super(KeyedDecoder, self).__init__(hidden_dim, embedding_dim, output_embedding_matrix, num_words_outputs,
                                           pad_idx, dropout)
        self.gru = KeyedGRU(embedding_dim, hidden_dim, keyed_kwargs)
        self.register_buffer('key', keyed_kwargs.get('dec_key', torch.randint(0, num_words_outputs, (1, 15))))
        self.key_emb = None

    def set_key(self, key):
        self.register_buffer('key', key)
        self.key_emb = None

    def reset_key_emb(self):
        self.key_emb = self.output_embedding(self.key)

    def get_signature(self, reduce=True):
        if reduce:
            return self.gru.get_key_gate(self.output_embedding(self.key), return_first_hidden=True)[1].mean(dim=0)
        else:
            return self.gru.get_key_gate(self.output_embedding(self.key), return_first_hidden=True)[1]

    def forward(self, x, hidden, use_key=True, key_hidden=None, timestep=0):
        embedded = self.output_embedding(x)
        # decoder forward is implemented in timestep fashion
        if use_key:
            if timestep < self.key_emb.size()[1]:
                dec_hidden, key_hidden = self.gru(embedded, key=self.key_emb[:, timestep], init_hidden=hidden,
                                                  teacher_forcing=False, key_hidden=key_hidden, timestep=timestep)
                output = self.softmax(self.classifier(self.dropout(dec_hidden)))
                return output, dec_hidden, key_hidden
            else:
                dec_hidden, key_hidden = self.gru(embedded, init_hidden=hidden, teacher_forcing=False)
                output = self.softmax(self.classifier(self.dropout(dec_hidden)))
                return output, dec_hidden, key_hidden
        else:
            dec_hidden, _ = self.gru(embedded, init_hidden=hidden, teacher_forcing=False)
            output = self.softmax(self.classifier(self.dropout(dec_hidden)))
            return output, dec_hidden
