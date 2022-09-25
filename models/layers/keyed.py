import torch
import torch.nn as nn
import torch.nn.init as init

from models.losses.sign_loss import SignLoss
from models.layers.util import str_to_sign


class KeyedLayer:
    # parent class used to identify instance of keyed layer
    pass


class KeyedLSTM(nn.Module, KeyedLayer):
    def __init__(self, input_size, hidden_size, keyed_kwargs=None, hidden_init='zero', return_sequences=False,
                 return_state=False, batch_first=True, bias=True):

        super(KeyedLSTM, self).__init__()

        if keyed_kwargs is None:
            keyed_kwargs = {}
            print('warning empty keyed_kwargs')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_zero = hidden_init == 'zero'
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bias = bias
        self.batch_first = batch_first
        self.cell = nn.LSTMCell(input_size, hidden_size, bias)
        self.alpha = keyed_kwargs.get('sign_loss_alpha', 1)
        signature = keyed_kwargs.get('signature', torch.sign(torch.rand(hidden_size) - 0.5))
        if isinstance(signature, int):
            signature = torch.ones(hidden_size) * signature
        elif isinstance(signature, str):
            signature = torch.FloatTensor(str_to_sign(signature, hidden_size))

        self.register_buffer('signature', signature)

        if self.alpha != 0:
            self.sign_loss = SignLoss(self.alpha, self.signature, keyed_kwargs.get('sign_std_alpha', 0.001))
        else:
            self.sign_loss = None

    def get_key_gate(self, key, key_hidden=None, return_first_hidden=False):
        # get key gate with the given key
        def one_step(x, hidden):
            # one timestep operation of the lstmcell
            self.cell.check_forward_input(x)
            hx, cx = hidden
            self.cell.check_forward_hidden(x, hx, '[0]')
            self.cell.check_forward_hidden(x, cx, '[1]')
            gates = torch.mm(x, self.cell.weight_ih.t()) + torch.mm(hx,
                                                                    self.cell.weight_hh.t()) + self.cell.bias_ih + self.cell.bias_hh
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            return (hy, cy), forgetgate

        if key_hidden is None:
            # get key gate for all timestep of the given key
            key = key.permute(1, 0, 2)
            key_batch = key.size()[1]
            key_hx = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
            key_cx = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
            if not self.init_zero:
                init.xavier_normal(key_hx)
                init.xavier_normal(key_cx)
            key_final_out = []
            key_hidden = (key_hx, key_cx)
            for i in range(key.size()[0]):
                key_hidden, keygate = one_step(key[i], key_hidden)
                if i == 0 and return_first_hidden:
                    first_hidden = key_hidden
                key_final_out.append(keygate)
            if return_first_hidden:
                return key_final_out, first_hidden
            return key_final_out
        else:
            # get key gate for one timestep
            return one_step(key, key_hidden)

    def forward(self, x, key=None, init_hidden=None, teacher_forcing=True, timestep=None, key_hidden=None):
        if not teacher_forcing:
            # if teacher_forcing is False, then the layer is used in a for loop in each timestep
            assert init_hidden is not None
            # input in batch of sos token (batch size, input size)
            hx, cx = self.cell(x, init_hidden)
            key_init_hidden = key_hidden

            if key is not None:
                key_batch = key.size()[0]
                if key_init_hidden is None or timestep == 0:
                    # initialize key hidden state
                    key_hx = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
                    key_cx = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
                    if not self.init_zero:
                        init.xavier_normal(key_hx)
                        init.xavier_normal(key_cx)
                    key_hidden = (key_hx, key_cx)
                # get key gate for one timestep
                key_hidden, key_gate = self.get_key_gate(key, key_hidden)

                hx = hx * key_gate.mean(dim=0)
                cx = cx * key_gate.mean(dim=0)

                if self.sign_loss is not None and (key_init_hidden is None or timestep == 0):
                    self.sign_loss.reset()
                    self.sign_loss.add(key_hidden[0])

            return (hx, cx), key_hidden
        else:
            if self.batch_first:
                # input in batch size, time step, input size
                x = x.permute(1, 0, 2)
            batch_size = x.size()[1]
            if init_hidden is not None:
                hx, cx = init_hidden
            else:
                hx = torch.zeros((batch_size, self.hidden_size), device=self.cell.weight_hh.device)
                cx = torch.zeros((batch_size, self.hidden_size), device=self.cell.weight_ih.device)
                if not self.init_zero:
                    init.xavier_normal(hx)
                    init.xavier_normal(cx)
            if self.return_sequences:
                output = []

            if key is not None:
                key_gates, key_hidden = self.get_key_gate(key, return_first_hidden=True)
                key_len = len(key_gates)

            for i in range(x.size()[0]):
                hx, cx = self.cell(x[i], (hx, cx))
                if self.return_sequences:
                    output.append(hx)
                if key is not None:
                    if i < key_len:
                        hx = hx * key_gates[i].mean(dim=0)
                        cx = cx * key_gates[i].mean(dim=0)

            if self.sign_loss is not None and key is not None:
                self.sign_loss.reset()
                self.sign_loss.add(key_hidden[0])

            if self.return_sequences:
                output = torch.stack(output, dim=0)
            else:
                output = hx
            if self.return_state:
                return output, (hx, cx)
            else:
                return output


class KeyedGRU(nn.Module, KeyedLayer):
    def __init__(self, input_size, hidden_size, keyed_kwargs=None, hidden_init='zero', return_sequences=False,
                 return_state=False, batch_first=True, bias=True):

        super(KeyedGRU, self).__init__()

        if keyed_kwargs is None:
            keyed_kwargs = {}
            print('warning empty keyed_kwargs')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_zero = hidden_init == 'zero'
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bias = bias
        self.batch_first = batch_first
        self.cell = nn.GRUCell(input_size, hidden_size, bias)
        self.alpha = keyed_kwargs.get('sign_loss_alpha', 1)
        signature = keyed_kwargs.get('signature', torch.sign(torch.rand(hidden_size) - 0.5))
        if isinstance(signature, int):
            signature = torch.ones(hidden_size) * signature
        elif isinstance(signature, str):
            signature = torch.FloatTensor(str_to_sign(signature, hidden_size))

        self.register_buffer('signature', signature)

        if self.alpha != 0:
            self.sign_loss = SignLoss(self.alpha, self.signature, keyed_kwargs.get('sign_std_alpha', 0.001))
        else:
            self.sign_loss = None

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def get_key_gate(self, key, key_hidden=None, return_first_hidden=False):
        # get key gate with the given key
        def one_step(x, hidden):
            # one timestep operation of the lstmcell
            self.cell.check_forward_input(x)
            self.cell.check_forward_hidden(x, hidden, '')
            gi = torch.mm(x, self.cell.weight_ih.t()) + self.cell.bias_ih
            gh = torch.mm(hidden, self.cell.weight_hh.t()) + self.cell.bias_hh
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            return hy, resetgate

        if key_hidden is None:
            # get key gate for all timestep of the given key
            key = key.permute(1, 0, 2)
            key_batch = key.size()[1]
            key_hidden = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
            if not self.init_zero:
                init.xavier_normal(key_hidden)
            key_final_out = []
            for i in range(key.size()[0]):
                key_hidden, keygate = one_step(key[i], key_hidden)
                if i == 0 and return_first_hidden:
                    first_hidden = key_hidden
                key_final_out.append(keygate)
            if return_first_hidden:
                return key_final_out, first_hidden
            return key_final_out
        else:
            # get key gate for one timestep
            return one_step(key, key_hidden)

    def forward(self, x, key=None, init_hidden=None, teacher_forcing=True, timestep=None, key_hidden=None):
        if not teacher_forcing:
            # if teacher_forcing is False, then the layer is used in a for loop in each timestep
            assert init_hidden is not None
            # input in batch of single token (batch size, input size)
            hx = self.cell(x, init_hidden)
            key_init_hidden = key_hidden

            if key is not None:
                key_batch = key.size()[0]
                if key_init_hidden is None or timestep == 0:
                    # initialize key hidden state
                    key_hidden = torch.zeros((key_batch, self.hidden_size), device=self.cell.weight_hh.device)
                    if not self.init_zero:
                        init.xavier_normal(key_hidden)
                # get key gate for one timestep
                key_hidden, key_gate = self.get_key_gate(key, key_hidden)
                hx = hx * key_gate.mean(dim=0)

                if self.sign_loss is not None and (key_init_hidden is None or timestep == 0):
                    self.sign_loss.reset()
                    self.sign_loss.add(key_hidden)

            return hx, key_hidden
        else:
            if self.batch_first:
                # input in batch size, time step, input size
                x = x.permute(1, 0, 2)
            batch_size = x.size()[1]
            if init_hidden is not None:
                hx = init_hidden
            else:
                hx = torch.zeros((batch_size, self.hidden_size), device=self.cell.weight_hh.device)
                if not self.init_zero:
                    init.xavier_normal(hx)
            if self.return_sequences:
                output = []

            if key is not None:
                key_gates, key_hidden = self.get_key_gate(key, return_first_hidden=True)
                key_len = len(key_gates)

            for i in range(x.size()[0]):
                hx = self.cell(x[i], hx)
                if self.return_sequences:
                    output.append(hx)
                if key is not None:
                    if i < key_len:
                        hx = hx * key_gates[i].mean(dim=0)

            if self.sign_loss is not None and key is not None:
                self.sign_loss.reset()
                self.sign_loss.add(key_hidden)

            if self.return_sequences:
                output = torch.stack(output, dim=0)
            else:
                output = hx
            if self.return_state:
                return output, hx
            else:
                return output
