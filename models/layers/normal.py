import torch
import torch.nn as nn
import torch.nn.init as init


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_init='zero', return_sequences=False, return_state=False,
                 batch_first=True, bias=True, dropout=0):

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_zero = hidden_init == 'zero'
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.cell = nn.LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, init_hidden=None, teacher_forcing=True):
        if not teacher_forcing:
            assert init_hidden is not None
            # input in batch of sos token (batch size, input size)
            return self.cell(x, init_hidden)
        else:
            if self.batch_first:
                # input in batch size, time step, input size
                x = x.permute(1, 0, 2)
            batch_size = x.size()[1]

            if init_hidden is not None:
                hx, cx = init_hidden
            else:
                hx = torch.zeros((batch_size, self.hidden_size)).to(self.cell.weight_hh.device)
                cx = torch.zeros((batch_size, self.hidden_size)).to(self.cell.weight_ih.device)
                if not self.init_zero:
                    init.xavier_normal(hx)
                    init.xavier_normal(cx)
            if self.return_sequences:
                output = []

            for i in range(x.size()[0]):
                hx, cx = self.cell(x[i], (hx, cx))
                if self.return_sequences:
                    output.append(hx)
            if self.return_sequences:
                output = torch.stack(output, dim=0)
            else:
                output = hx
            if self.return_state:
                return output, (hx, cx)
            else:
                return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_init='zero', return_sequences=False, return_state=False,
                 batch_first=True, bias=True, dropout=0):

        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_zero = hidden_init == 'zero'
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.cell = nn.GRUCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, init_hidden=None, teacher_forcing=True):
        if not teacher_forcing:
            assert init_hidden is not None
            # input in batch of sos token (batch size, input size)
            return self.cell(x, init_hidden)
        else:
            if self.batch_first:
                # input in batch size, time step, input size
                x = x.permute(1, 0, 2)
            batch_size = x.size()[1]
            if init_hidden is not None:
                hx = init_hidden
            else:
                hx = torch.zeros((batch_size, self.hidden_size)).to(self.cell.weight_hh.device)
                if not self.init_zero:
                    init.xavier_normal(hx)
            if self.return_sequences:
                output = []
            for i in range(x.size()[0]):
                hx = self.cell(x[i], hx)
                if self.return_sequences:
                    output.append(hx)
            if self.return_sequences:
                output = torch.stack(output, dim=0)
            else:
                output = hx
            if self.return_state:
                return output, hx
            else:
                return output
