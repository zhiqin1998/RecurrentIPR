import random
import numpy as np


def str_to_sign(s, dim=1024):
    # function to convert string to its sign signature
    s = bin(int.from_bytes(s.encode(), 'big')).replace('b', '')
    s = [1 if c == '1' else -1 for c in s]
    if len(s) >= dim:
        print('warning string length * 8 is larger than weight dimension {}'.format(dim))
        return s[:dim]
    while len(s) != dim:
        s.append(1 if bool(random.getrandbits(1)) else -1)
    return s


def sign_to_str(sign, str_len):
    # reverse function to convert sign signature back to string
    sign = sign[:str_len * 8]
    s = ''.join(['1' if x > 0 else '0' for x in sign])
    n = int(s, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()


def calculate_ber(sign, string):
    # calculate bit error rate of sign and the string signature
    ts = bin(int.from_bytes(string.encode(), 'big')).replace('b', '')
    ts = [1 if c == '1' else 0 for c in ts]
    sign = sign[:len(string) * 8]
    gts = [1 if x > 0 else 0 for x in sign]
    return 1 - sum(x == y for x, y in zip(gts, ts)) / len(gts)


def random_flip_sign(weight, perc=0.5):
    randomidx = np.random.choice(len(weight), int(len(weight) * perc), replace=False)
    weight[randomidx] = weight[randomidx] * -1
    return weight
