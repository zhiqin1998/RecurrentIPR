import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLoss(nn.Module):
    def __init__(self, alpha, signature=None, std_alpha=0.001, min_val=0.1, regularize=True):
        super(SignLoss, self).__init__()
        self.alpha = alpha
        self.std_alpha = std_alpha
        self.register_buffer('signature', signature)
        self.regularize = regularize
        self.min_val = min_val
        self.loss = 0
        self.acc = 0
        self.kh_cache = None

    def set_signature(self, signature):
        self.signature.copy_(signature)

    def get_acc(self):
        if self.kh_cache is not None:
            acc = (torch.sign(self.signature.view(-1)) == torch.sign(self.kh_cache.view(-1))).float().mean()
            return acc
        else:
            raise Exception('kh_cache is None')

    def get_loss(self):
        if self.kh_cache is not None:
            loss = (self.alpha * F.relu(-self.signature.view(-1) * self.kh_cache.view(-1) + self.min_val)).sum()
            return loss
        else:
            raise Exception('scale_cache is None')

    def add(self, key_hidden):
        key_hidden = key_hidden[:, :len(self.signature)]
        self.kh_cache = key_hidden.mean(dim=0)

        # hinge loss concept
        # f(x) = max(x + 0.5, 0)*-b
        # f(x) = max(x + 0.5, 0) if b = -1
        # f(x) = max(0.5 - x, 0) if b = 1

        # case b = -1
        # - (-1) * 1 = 1 === bad
        # - (-1) * -1 = -1 -> 0 === good

        # - (-1) * 0.6 + 0.5 = 1.1 === bad
        # - (-1) * -0.6 + 0.5 = -0.1 -> 0 === good

        # case b = 1
        # - (1) * -1 = 1 -> 1 === bad
        # - (1) * 1 = -1 -> 0 === good

        # let it has minimum of 0.1
        self.loss += self.get_loss()
        if key_hidden.size()[0] > 1:
            self.loss += (1 / (key_hidden.std(dim=0)).mean()) * self.std_alpha  # keeping std high
        if self.regularize:
            self.loss += (0.00001 * self.kh_cache.view(-1).pow(2).sum())  # to regularize the scale not to be so large
        self.acc += self.get_acc()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.kh_cache = None
