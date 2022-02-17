import numpy as np

from sklearn.utils import resample

from collections import defaultdict
from collections import namedtuple

mt_info = namedtuple("Info", ("target_bpe", "model_scores"))


def upsample(X, y):
    y = np.expand_dims(y, 1)
    data = np.hstack((X, y))
    majority = data[data[:, -1] == 1]
    minority = data[data[:, -1] == 0]
    minority = resample(minority, replace=True, n_samples=majority.shape[0], random_state=1234)
    data = np.concatenate([majority, minority])
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1].squeeze()


def get_statistics(X):
    mean_ = np.nanmean(X, 0)
    scale_ = np.nanstd(X, 0)
    return mean_, scale_


def scale(X, mean_, scale_):
    Xr = np.rollaxis(X, 0)
    Xr -= mean_
    scale_ = scale_.copy()
    scale_[scale_ == 0.0] = 1.0
    Xr /= scale_
    return X


def read_mt_info(fname):
    output = []
    for line in open(fname):
        parts = line.strip().split("|||")
        assert len(parts) == 5
        target_bpe = parts[1].lstrip()
        model_scores = list(map(float, parts[2].lstrip(" WordScores= ").split()))
        output.append(mt_info(target_bpe, model_scores))
    return output


def map_token_labels_to_word_labels(target_moses, target_words, labels):

    def map_word_to_chrts(words):
        word_to_chrts = defaultdict(list)
        word_idx = 0
        chrt_idx = 0
        for chrt in words:
            if chrt != " ":
                word_to_chrts[word_idx].append(chrt_idx)
                chrt_idx += 1
            else:
                word_idx += 1
        return word_to_chrts

    def map_chrt_to_label(tokens, labels):
        chrt_to_label = {}
        tok_idx = 0
        chrt_idx = 0
        for chrt in tokens:
            chrt_to_label[chrt_idx] = labels[tok_idx]
            if chrt == " ":
                tok_idx += 1
            else:
                chrt_idx += 1
        return chrt_to_label

    word_to_chrts = map_word_to_chrts(target_words)
    chrt_to_label = map_chrt_to_label(target_moses, labels)
    labels = []
    for i in range(len(target_words.split())):
        ls = []
        for chrt_idx in word_to_chrts[i]:
            ls.append(chrt_to_label[chrt_idx])
        labels.append(max(ls))
    return labels


def map_words_to_bpe_tokens(bpe_tokens, bpe_sep='▁'):
    """
    :param bpe_tokens: str
    :param bpe_sep: bpe separator
    :return: List[Tuple]
    Returns the a list where each element is a tuple containing the indexes of bpe tokens corresponding
    to each target_moses word
    """
    indexes = []
    word = []
    chr_idx = 0
    bpe_idx = 0
    bpe_tokens = bpe_tokens.lstrip(bpe_sep)
    while True:
        if chr_idx == len(bpe_tokens) - 1:
            word.append(bpe_idx)
            indexes.append(tuple(word))
            break
        if bpe_tokens[chr_idx] == bpe_sep:
            indexes.append(tuple(word))
            word = []
        elif bpe_tokens[chr_idx] == ' ':
            word.append(bpe_idx)
            bpe_idx += 1
        else:
            pass
        chr_idx += 1
    return indexes
