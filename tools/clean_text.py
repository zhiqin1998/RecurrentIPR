import re
import string
import spacy
import argparse


def is_num(s):
    return any(i.isdigit() for i in s)


def count_alpha(s):
    return sum([x.isalpha() for x in s])


def count_single(s):
    return sum([len(x) == 1 for x in s.split()])


if __name__ == '__main__':
    # script to cleanup parallel corpus from wmt14, use with
    # `python clean_text.py --src-file=corpus.en --trg-file=corpus.fr --src-out-file=cleaned.en --src-trg-file=cleaned.fr`
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', type=str, dest='src_file', default='./train/full_tokenized.en',
                        help='src file')
    parser.add_argument('--trg-file', type=str, dest='trg_file', default='./train/full_tokenized.fr',
                        help='trg file')
    parser.add_argument('--src-out-file', type=str, dest='out_src_file', default='./train/combined_tokenized.en',
                        help='src output file')
    parser.add_argument('--trg-out-file', type=str, dest='out_trg_file', default='./train/combined_tokenized.fr',
                        help='trg output file')
    parser.add_argument('--max-length', type=int, dest='max_sentence_length', default=20,
                        help='max sentence_length')
    args = parser.parse_args()
    spc = '#$%&()*+-/:;<=>@[\\]^_`{|}~'
    table = str.maketrans('', '', spc)
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    src_file = args.src_file
    trg_file = args.trg_file

    out_src_file = args.out_src_file
    out_trg_file = args.out_trg_file

    max_sentence_length = args.max_sentence_length
    mem = []
    trg_mem = []

    src_spacy = spacy.load(src_file.split('.')[-1])
    trg_spacy = spacy.load(trg_file.split('.')[-1])

    with open(src_file, encoding="utf-8", newline='\n') as src_f, open(trg_file, encoding="utf-8",
                                                                       newline='\n') as trg_f, open(out_src_file,
                                                                                                    'w') as out_src_f, open(
        out_trg_file, 'w') as out_trg_f:
        for src_sent, trg_sent in zip(src_f, trg_f):
            src_sent, trg_sent = src_sent.strip(), trg_sent.strip()

            # remove punctuation, non-printable character, and number
            src_sent = src_sent.translate(table)
            src_sent = re_print.sub('', src_sent)
            src_sent = ' '.join([w for w in src_sent.split() if not is_num(w)])

            trg_sent = trg_sent.replace('«', '"').replace('»', '"')
            trg_sent = trg_sent.translate(table)
            trg_sent = re_print.sub('', trg_sent)
            trg_sent = ' '.join([w for w in trg_sent.split() if not is_num(w)])

            ns_src_sent = ''.join(src_sent.split())
            ns_trg_sent = ''.join(trg_sent.split())

            # skip long sentence
            if len(src_sent.split()) > max_sentence_length or len(trg_sent.split()) > 1.5 * max_sentence_length:
                continue
            # filter repeating punctuation
            if re.search(r'\.{2,}', src_sent) or re.search(r',{2,}', src_sent) or re.search(r'!{2,}',
                                                                                            src_sent) or re.search(
                    r'"{2,}', src_sent) or re.search(r"'{2,}", src_sent):
                continue
            if re.search(r'\.{2,}', trg_sent) or re.search(r',{2,}', trg_sent) or re.search(r'!{2,}',
                                                                                            trg_sent) or re.search(
                    r'"{2,}', trg_sent) or re.search(r"'{2,}", trg_sent):
                continue
            if src_sent == '' or trg_sent == '' or src_sent == trg_sent:
                continue
            # skip non alphabetic sentence
            if count_alpha(ns_src_sent) / len(ns_src_sent) < 0.5 or count_alpha(ns_trg_sent) / len(ns_trg_sent) < 0.5:
                continue
            if count_single(src_sent) / len(ns_src_sent) > 0.5 or count_single(trg_sent) / len(ns_trg_sent) > 0.5:
                continue

            if src_sent in mem or trg_sent in trg_mem:
                # skip duplicates
                continue
            else:
                mem.append(src_sent)
                trg_mem.append(trg_sent)
                if len(mem) > 100:
                    mem.pop(0)
                if len(trg_mem) > 100:
                    trg_mem.pop(0)

            # tokenize with spacy
            src_sent = ' '.join([tok.text for tok in src_spacy.tokenizer(src_sent)])
            trg_sent = ' '.join([tok.text for tok in trg_spacy.tokenizer(trg_sent)])

            out_src_f.write(src_sent)
            out_src_f.write('\n')
            out_trg_f.write(trg_sent)
            out_trg_f.write('\n')
