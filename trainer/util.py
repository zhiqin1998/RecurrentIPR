import torch


def sequence_to_text(sequences, vocab, include_eos=False):
    # sequences shape batch size, seq_len
    output = []
    for i in range(sequences.size()[0]):
        temp = []
        for j in range(sequences.size()[1]):
            idx = sequences[i, j].item()
            if idx == vocab.stoi['<pad>']:
                continue
            elif idx == vocab.stoi['<eos>']:
                if include_eos:
                    temp.append('<eos>')
                break
            else:
                temp.append(vocab.itos[idx])
        output.append(temp)
    return output


def merge_sequence_batch(x, y, pad_idx):
    if len(x.size()) == 2:
        final = torch.empty((x.size()[0] + y.size()[0], max(x.size()[1], y.size()[1])), dtype=x.dtype).fill_(pad_idx)
        final[:x.size()[0], :x.size()[1]] = x
        final[x.size()[0]:, :y.size()[1]] = y
    else:
        final = torch.cat([x, y], dim=0)
    return final
