import torch
import torch.nn as nn

from models.layers.normal import LSTM, GRU
from models.layers.keyed import KeyedLSTM, KeyedGRU


class TestLSTMNormal(nn.Module):
    def __init__(self, input_size, hidden_size, output_class):
        super(TestLSTMNormal, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_class = output_class

        self.features = nn.Sequential(
            LSTM(input_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size, output_class)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestLSTMKeyed(nn.Module):
    def __init__(self, input_size, hidden_size, output_class, keyed_kwargs=None):
        super(TestLSTMKeyed, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_class = output_class

        self.lstm = KeyedLSTM(input_size, hidden_size, keyed_kwargs=keyed_kwargs)
        self.classifier = nn.Linear(hidden_size, output_class)

        self.register_buffer('key', keyed_kwargs.get('key', torch.randn((input_size, input_size))).detach().clone())

    def set_key(self, key):
        self.register_buffer('key', key.detach().clone())

    def forward(self, x, use_key=True):
        if use_key:
            x = self.lstm(x, key=self.key)
        else:
            x = self.lstm(x)
        x = self.classifier(x)
        return x


class TestBiLSTMKeyed(nn.Module):
    def __init__(self, input_size, hidden_size, output_class, keyed_kwargs=None):
        super(TestBiLSTMKeyed, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_class = output_class

        self.lstm = KeyedLSTM(input_size, hidden_size, keyed_kwargs=keyed_kwargs)
        self.classifier = nn.Linear(hidden_size * 2, output_class)

        self.register_buffer('key', keyed_kwargs.get('key', torch.randn((input_size, input_size))).detach().clone())

    def set_key(self, key):
        self.register_buffer('key', key.detach().clone())

    def forward(self, x, use_key=True):
        rev_x = torch.flip(x, (1,))
        if use_key:
            x = self.lstm(x, key=self.key)
            rev_x = self.lstm(rev_x, key=self.key)
        else:
            x = self.lstm(x)
            rev_x = self.lstm(rev_x, key=self.key)
        x = self.classifier(torch.cat((x, rev_x), 1))
        return x


class TestGRUNormal(nn.Module):
    def __init__(self, input_size, hidden_size, output_class):
        super(TestGRUNormal, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_class = output_class

        self.features = nn.Sequential(
            GRU(input_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size, output_class)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestGRUKeyed(nn.Module):
    def __init__(self, input_size, hidden_size, output_class, keyed_kwargs=None):
        super(TestGRUKeyed, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_class = output_class

        self.gru = KeyedGRU(input_size, hidden_size, keyed_kwargs=keyed_kwargs)
        self.classifier = nn.Linear(hidden_size, output_class)

        self.register_buffer('key', keyed_kwargs.get('key', torch.randn((input_size, input_size))).detach().clone())

    def set_key(self, key):
        self.register_buffer('key', key.detach().clone())

    def forward(self, x, use_key=True):
        if use_key:
            x = self.gru(x, key=self.key)
        else:
            x = self.gru(x)
        x = self.classifier(x)
        return x
