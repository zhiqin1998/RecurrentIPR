import time
import random
import torch

from models.losses.sign_loss import SignLoss
from models.rnn import KeyedBiRNN
from trainer.util import merge_sequence_batch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class Trainer:
    def __init__(self, model, optimizer, criterion, device, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip

    def train(self, e, dataloader, use_key=False):
        self.model.train()
        sign_loss_meter = 0
        loss_meter = 0
        acc_meter = 0

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # reset sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            if isinstance(self.model, KeyedBiRNN):
                pred = self.model(data, use_key=use_key)
            else:
                pred = self.model(data)
            loss = self.criterion(pred, target)
            sign_loss = torch.tensor(0.).to(self.device)

            # add up sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss

            (loss + sign_loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            sign_loss_meter = sign_loss.item()
            loss_meter += loss.item()
            acc_meter += accuracy(pred, target)[0].item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Sign Loss: {sign_loss_meter / (i + 1):6.4f} '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

        print()

        loss_meter /= len(dataloader)
        acc_meter /= len(dataloader)

        sign_acc = torch.tensor(0.).to(self.device)
        count = 0

        for m in self.model.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1

        if count != 0:
            sign_acc /= count

        return {'loss': loss_meter,
                'sign_loss': sign_loss_meter,
                'sign_acc': sign_acc.item(),
                'acc': acc_meter,
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing Result', use_key=False):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0

        start_time = time.time()
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                target = target.to(self.device)

                if isinstance(self.model, KeyedBiRNN):
                    pred = self.model(data, use_key=use_key)
                else:
                    pred = self.model(data)
                loss = self.criterion(pred, target)

                loss_meter += loss.item()
                acc_meter += accuracy(pred, target)[0].item()

        loss_meter /= len(dataloader)
        acc_meter /= len(dataloader)
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}


class PrivateTrainer:
    def __init__(self, model, optimizer, criterion, device, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip

    def train(self, e, dataloader, trigger_dataloader=None, pad_idx=1):
        self.model.train()
        sign_loss_meter = 0
        public_loss_meter = 0
        private_loss_meter = 0
        public_acc_meter = 0
        private_acc_meter = 0

        if trigger_dataloader:
            # load and trigger dataloader
            t_datas, t_targets = [], []
            for data, target in trigger_dataloader:
                t_datas.append(data)
                t_targets.append(target)
            t_idx = random.randrange(len(t_datas))

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):

            if trigger_dataloader:
                # merge trigger batch into training batch
                t_data = t_datas[(t_idx + i) % len(t_datas)]
                t_target = t_targets[(t_idx + i) % len(t_targets)]

                data = merge_sequence_batch(data, t_data, pad_idx)
                target = torch.cat([target, t_target], dim=0)

            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # reset sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            # two forward pass with (public) and without key (private)
            pred = self.model(data, use_key=False)
            public_loss = self.criterion(pred, target)
            public_acc_meter += accuracy(pred, target)[0].item()

            pred = self.model(data, use_key=True)
            private_loss = self.criterion(pred, target)
            private_acc_meter += accuracy(pred, target)[0].item()

            sign_loss = torch.tensor(0.).to(self.device)

            # add up sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss

            # combine loss (joint learning)
            (public_loss + private_loss + sign_loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            sign_loss_meter = sign_loss.item()
            public_loss_meter += public_loss.item()
            private_loss_meter += private_loss.item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Sign Loss: {sign_loss_meter / (i + 1):6.4f} '
                  f'Public Loss: {public_loss_meter / (i + 1):6.4f} '
                  f'Public Acc: {public_acc_meter / (i + 1):.4f} '
                  f'Private Loss: {private_loss_meter / (i + 1):6.4f} '
                  f'Private Acc: {private_acc_meter / (i + 1):.4f} '
                  f'({time.time() - start_time:.2f}s)', end='\r')

        print()
        sign_loss_meter /= len(dataloader)
        public_acc_meter /= len(dataloader)
        private_acc_meter /= len(dataloader)
        public_loss_meter /= len(dataloader)
        private_loss_meter /= len(dataloader)

        sign_acc = torch.tensor(0.).to(self.device)
        count = 0

        for m in self.model.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1

        if count != 0:
            sign_acc /= count

        return {'public_loss': public_loss_meter,
                'private_loss': private_loss_meter,
                'public_acc': public_acc_meter,
                'private_acc': private_acc_meter,
                'sign_loss': sign_loss_meter,
                'sign_acc': sign_acc.item(),
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing Result'):
        self.model.eval()
        public_loss_meter = 0
        private_loss_meter = 0
        public_acc_meter = 0
        private_acc_meter = 0

        start_time = time.time()
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data, use_key=False)
                public_loss = self.criterion(pred, target)
                public_loss_meter += public_loss.item()
                public_acc_meter += accuracy(pred, target)[0].item()

                pred = self.model(data, use_key=True)
                private_loss = self.criterion(pred, target)
                private_loss_meter += private_loss.item()
                private_acc_meter += accuracy(pred, target)[0].item()

        public_acc_meter /= len(dataloader)
        private_acc_meter /= len(dataloader)
        public_loss_meter /= len(dataloader)
        private_loss_meter /= len(dataloader)

        print(f'{msg}: '
              f'Public Loss: {public_loss_meter:6.4f} '
              f'Private Loss: {private_loss_meter:6.4f} '
              f'Public Acc: {public_acc_meter:6.2f} '
              f'Private Acc: {private_acc_meter:6.2f} '
              f'({time.time() - start_time:.2f}s)')
        print()

        return {'public_loss': public_loss_meter,
                'private_loss': private_loss_meter,
                'public_acc': public_acc_meter,
                'private_acc': private_acc_meter,
                'time': time.time() - start_time}
