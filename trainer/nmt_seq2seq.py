import random
import time
import torch

from models.losses.sign_loss import SignLoss
from models.nmt_seq2seq import KeyedDecoder, KeyedEncoder
from trainer.util import sequence_to_text, merge_sequence_batch
from torchtext.data.metrics import bleu_score


class EncDecEvaluator:
    # evaluator to evaluate bleu score on test set
    def __init__(self, encoder, decoder, device, trg_vocab, teacher_forcing=1.0):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.trg_vocab = trg_vocab
        self.teacher_forcing = teacher_forcing

    def evaluate_bleu(self, dataloader, use_key=False):
        self.encoder.eval()
        self.decoder.eval()
        references = []
        hypothesis = []

        start_time = time.time()
        with torch.no_grad():
            for src, trg, target in dataloader:
                src = src.to(self.device)
                trg = trg.to(self.device)
                target = target.to(self.device)

                if isinstance(self.encoder, KeyedEncoder):
                    enc_hidden = self.encoder(src, use_key=use_key)
                else:
                    enc_hidden = self.encoder(src)

                seq_len = trg.size()[1]
                batch_size = trg.size()[0]
                pred = torch.zeros((seq_len, batch_size), device=self.device, dtype=torch.int32)

                if use_key:
                    self.decoder.reset_key_emb()
                    dec_hidden = enc_hidden
                    key_hidden = None
                    dec_in = trg[:, 0]

                    for si in range(seq_len):
                        dec_out, dec_hidden, key_hidden = self.decoder(dec_in, dec_hidden, use_key=use_key,
                                                                       key_hidden=key_hidden, timestep=si)
                        pred[si] = dec_out.argmax(dim=1)
                        if si + 1 < seq_len:
                            dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)
                else:
                    keyed_dec = isinstance(self.decoder, KeyedDecoder)
                    dec_hidden = enc_hidden
                    dec_in = trg[:, 0]

                    for si in range(seq_len):
                        if keyed_dec:
                            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden, use_key=False)
                        else:
                            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
                        pred[si] = dec_out.argmax(dim=1)
                        if si + 1 < seq_len:
                            dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)

                pred = pred.permute(1, 0)  # permute axis back to (batch size, time steps)
                # convert sequence back to text
                hypo = sequence_to_text(pred, self.trg_vocab)
                ref = sequence_to_text(target, self.trg_vocab)
                hypothesis.extend(hypo)
                for r in ref:
                    references.append([r])
        bleu_scores = bleu_score(hypothesis, references) * 100
        print('BLEU score: {:.4f}\t Time taken: {:.4f}s'.format(bleu_scores, time.time() - start_time))
        return {'bleu_score': bleu_scores,
                'references': references,
                'hypothesis': hypothesis}


class EncDecTrainer:
    def __init__(self, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device, grad_clip=None,
                 teacher_forcing=0.5):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.teacher_forcing = teacher_forcing

    def train(self, e, dataloader, use_key=False):
        self.encoder.train()
        self.decoder.train()
        sign_loss_meter = 0
        loss_meter = 0

        start_time = time.time()
        for i, (src, trg, target) in enumerate(dataloader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            target = target.to(self.device)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            loss = torch.tensor(0.).to(self.device)

            # reset sign loss in model
            for m in self.encoder.modules():
                if isinstance(m, SignLoss):
                    m.reset()
            for m in self.decoder.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            if isinstance(self.encoder, KeyedEncoder):
                enc_hidden = self.encoder(src, use_key=use_key)
            else:
                enc_hidden = self.encoder(src)

            seq_len = trg.size()[1]

            if use_key:
                # recalculate embedding for key in decoder model
                self.decoder.reset_key_emb()
                dec_hidden = enc_hidden
                key_hidden = None
                dec_in = trg[:, 0]

                for si in range(seq_len):
                    dec_out, dec_hidden, key_hidden = self.decoder(dec_in, dec_hidden, use_key=use_key,
                                                                   key_hidden=key_hidden, timestep=si)
                    loss += self.criterion(dec_out, target[:, si])
                    if si + 1 < seq_len:
                        dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(
                            dim=1).detach()
            else:
                keyed_dec = isinstance(self.decoder, KeyedDecoder)
                dec_hidden = enc_hidden
                dec_in = trg[:, 0]

                for si in range(seq_len):
                    if keyed_dec:
                        dec_out, dec_hidden = self.decoder(dec_in, dec_hidden, use_key=False)
                    else:
                        dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
                    loss += self.criterion(dec_out, target[:, si])
                    if si + 1 < seq_len:
                        dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(
                            dim=1).detach()

            loss /= seq_len  # average loss over time steps
            sign_loss = torch.tensor(0.).to(self.device)

            for m in self.encoder.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss
            for m in self.decoder.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss

            (loss + sign_loss).backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            sign_loss_meter += sign_loss.item()
            loss_meter += loss.item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Sign Loss: {sign_loss_meter / (i + 1):6.4f} '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'({time.time() - start_time:.2f}s)', end='\r')
        print()

        loss_meter /= len(dataloader)
        sign_loss_meter /= len(dataloader)

        sign_acc = torch.tensor(0.).to(self.device)
        count = 0

        for m in self.encoder.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1
        for m in self.decoder.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1

        if count != 0:
            sign_acc /= count

        return {'loss': loss_meter,
                'sign_loss': sign_loss_meter,
                'sign_acc': sign_acc.item(),
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing', use_key=False):
        self.encoder.eval()
        self.decoder.eval()
        loss_meter = 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                src, trg, target = load
                src = src.to(self.device)
                trg = trg.to(self.device)
                target = target.to(self.device)

                loss = torch.tensor(0.).to(self.device)

                if isinstance(self.encoder, KeyedEncoder):
                    enc_hidden = self.encoder(src, use_key=use_key)
                else:
                    enc_hidden = self.encoder(src)
                    
                seq_len = trg.size()[1]

                if use_key:
                    self.decoder.reset_key_emb()
                    dec_hidden = enc_hidden
                    key_hidden = None
                    dec_in = trg[:, 0]

                    for si in range(seq_len):
                        dec_out, dec_hidden, key_hidden = self.decoder(dec_in, dec_hidden, use_key=use_key,
                                                                       key_hidden=key_hidden, timestep=si)
                        loss += self.criterion(dec_out, target[:, si])
                        if si + 1 < seq_len:
                            dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)
                    loss_meter += loss.item() / seq_len
                else:
                    keyed_dec = isinstance(self.decoder, KeyedDecoder)
                    dec_hidden = enc_hidden
                    dec_in = trg[:, 0]

                    for si in range(seq_len):
                        if keyed_dec:
                            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden, use_key=False)
                        else:
                            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
                        loss += self.criterion(dec_out, target[:, si])
                        if si + 1 < seq_len:
                            dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)
                    loss_meter += loss.item() / seq_len

        loss_meter /= len(dataloader)
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'({time.time() - start_time:.2f}s)')
        print()

        return {'loss': loss_meter,
                'time': time.time() - start_time}


class EncDecPrivateTrainer:
    def __init__(self, encoder, decoder, enc_optimizer, dec_optimizer, criterion, device, grad_clip=None,
                 teacher_forcing=0.5):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.teacher_forcing = teacher_forcing

    def train(self, e, dataloader, trigger_dataloader=None, reverse_input=True, pad_idx=1):
        self.encoder.train()
        self.decoder.train()
        sign_loss_meter = 0
        public_loss_meter = 0
        private_loss_meter = 0

        if trigger_dataloader:
            # load and trigger dataloader
            t_srcs, t_trgs, t_targets = [], [], []
            for src, trg, target in trigger_dataloader:
                if reverse_input:
                    src = torch.flip(src, (-1,))
                t_srcs.append(src)
                t_trgs.append(trg)
                t_targets.append(target)
            t_idx = random.randrange(len(t_srcs))

        start_time = time.time()
        for i, (src, trg, target) in enumerate(dataloader):

            if trigger_dataloader:
                # merge trigger batch into training batch
                if reverse_input:
                    src = torch.flip(src, (-1,))

                t_src = t_srcs[(t_idx + i) % len(t_srcs)]
                t_trg = t_trgs[(t_idx + i) % len(t_trgs)]
                t_target = t_targets[(t_idx + i) % len(t_targets)]

                if reverse_input:
                    src = torch.flip(merge_sequence_batch(src, t_src, pad_idx), (-1,))
                else:
                    src = merge_sequence_batch(src, t_src, pad_idx)
                trg = merge_sequence_batch(trg, t_trg, pad_idx)
                target = merge_sequence_batch(target, t_target, pad_idx)

            src = src.to(self.device)
            trg = trg.to(self.device)
            target = target.to(self.device)

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            # reset signloss of model
            for m in self.encoder.modules():
                if isinstance(m, SignLoss):
                    m.reset()
            for m in self.decoder.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            public_loss = torch.tensor(0.).to(self.device)  # public loss is forward pass without the key ( normal )
            enc_hidden = self.encoder(src, use_key=False)

            seq_len = trg.size()[1]

            dec_hidden = enc_hidden
            dec_in = trg[:, 0]

            for si in range(seq_len):
                dec_out, dec_hidden = self.decoder(dec_in, dec_hidden, use_key=False)
                public_loss += self.criterion(dec_out, target[:, si])
                if si + 1 < seq_len:
                    dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(
                        dim=1).detach()

            public_loss /= seq_len

            private_loss = torch.tensor(0.).to(self.device)  # private loss is forward pass with the key
            enc_hidden = self.encoder(src, use_key=True)

            self.decoder.reset_key_emb()
            dec_hidden = enc_hidden
            key_hidden = None
            dec_in = trg[:, 0]

            for si in range(seq_len):
                dec_out, dec_hidden, key_hidden = self.decoder(dec_in, dec_hidden, use_key=True, key_hidden=key_hidden,
                                                               timestep=si)
                private_loss += self.criterion(dec_out, target[:, si])
                if si + 1 < seq_len:
                    dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(
                        dim=1).detach()

            private_loss /= seq_len

            sign_loss = torch.tensor(0.).to(self.device)

            for m in self.encoder.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss
            for m in self.decoder.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss

            # combine all loss (joint learning)
            (public_loss + private_loss + sign_loss).backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

            self.enc_optimizer.step()
            self.dec_optimizer.step()

            sign_loss_meter += sign_loss.item()
            public_loss_meter += public_loss.item()
            private_loss_meter += private_loss.item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Sign Loss: {sign_loss_meter / (i + 1):6.4f} '
                  f'Public Loss: {public_loss_meter / (i + 1):6.4f} '
                  f'Private Loss: {private_loss_meter / (i + 1):6.4f} '
                  f'({time.time() - start_time:.2f}s)', end='\r')
        print()

        public_loss_meter /= len(dataloader)
        private_loss_meter /= len(dataloader)
        sign_loss_meter /= len(dataloader)

        sign_acc = torch.tensor(0.).to(self.device)
        count = 0

        # get sign accuracy
        for m in self.encoder.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1
        for m in self.decoder.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1

        if count != 0:
            sign_acc /= count

        return {'public_loss': public_loss_meter,
                'private_loss': private_loss_meter,
                'sign_loss': sign_loss_meter,
                'sign_acc': sign_acc.item(),
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing'):
        self.encoder.eval()
        self.decoder.eval()
        public_loss_meter = 0
        private_loss_meter = 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                src, trg, target = load
                src = src.to(self.device)
                trg = trg.to(self.device)
                target = target.to(self.device)

                public_loss = torch.tensor(0.).to(self.device)

                enc_hidden = self.encoder(src, use_key=False)

                seq_len = trg.size()[1]

                dec_hidden = enc_hidden
                dec_in = trg[:, 0]

                for si in range(seq_len):
                    dec_out, dec_hidden = self.decoder(dec_in, dec_hidden, use_key=False)
                    public_loss += self.criterion(dec_out, target[:, si])
                    if si + 1 < seq_len:
                        dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)
                public_loss_meter += public_loss.item() / seq_len

                private_loss = torch.tensor(0.).to(self.device)

                enc_hidden = self.encoder(src, use_key=True)

                self.decoder.reset_key_emb()
                dec_hidden = enc_hidden
                key_hidden = None
                dec_in = trg[:, 0]

                for si in range(seq_len):
                    dec_out, dec_hidden, key_hidden = self.decoder(dec_in, dec_hidden, use_key=True,
                                                                   key_hidden=key_hidden, timestep=si)
                    private_loss += self.criterion(dec_out, target[:, si])
                    if si + 1 < seq_len:
                        dec_in = trg[:, si + 1] if random.random() < self.teacher_forcing else dec_out.argmax(dim=1)
                private_loss_meter += private_loss.item() / seq_len

        public_loss_meter /= len(dataloader)
        private_loss_meter /= len(dataloader)
        print(f'{msg}: '
              f'Public Loss: {public_loss_meter:6.4f} '
              f'Private Loss: {private_loss_meter:6.4f} '
              f'({time.time() - start_time:.2f}s)')
        print()

        return {'public_loss': public_loss_meter,
                'private_loss': private_loss_meter,
                'time': time.time() - start_time}
