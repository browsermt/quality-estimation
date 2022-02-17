import numpy as np

from quality_estimation.data.utils import read_mt_info, map_words_to_bpe_tokens, map_token_labels_to_word_labels
from quality_estimation.data.data import InputData


class Dataset:

    def __init__(self, shuffle, retain_eos=False):
        """
        Dataset for storing glass-box data for quality estimation
        :param shuffle: bool
        :param scale: bool
        """
        self.shuffle = shuffle
        self.retain_eos = retain_eos
        self.data = []

    def collate_fn(self, indices):
        items = []
        labels = []
        for idx in indices:
            item, item_labels = self.data[idx]
            X = self.make_sentence_features(item)
            assert X.shape[0] == len(item_labels)
            items.append(X)
            labels.extend(item_labels)
        F = np.concatenate(items)
        labels = np.asarray(labels, dtype=np.int64)
        return F, labels

    def make_sentence_features(self, item):
        words_to_bpe = map_words_to_bpe_tokens(item.target_bpe)
        X = np.ndarray((len(words_to_bpe), 4))
        for word_index, bpe_indices in enumerate(words_to_bpe):
            bpe_scores = [item.model_scores[bpe_idx] for bpe_idx in bpe_indices]
            X[word_index, 0] = sum(bpe_scores) / len(bpe_scores)
            X[word_index, 1] = min(bpe_scores)
            X[word_index, 2] = len(bpe_scores)
            X[word_index, 3] = sum(item.model_scores) / len(item.model_scores)
        return X

    def read_data(self, path_src, path_mt, path_mt_info, path_labels):
        def _read_text(path):
            out = []
            with open(path, "r", encoding="utf-8") as f:
                for line in open(path):
                    out.append(line.strip())
            return out
        mt_infos = read_mt_info(path_mt_info)
        src = _read_text(path_src)
        mt = _read_text(path_mt)
        labels = _read_text(path_labels)
        assert len(src) == len(mt_infos) == len(mt) == len(labels)
        for i in range(len(src)):
            if labels[i] == "###NOT ANNOTATED###":
                continue
            item = InputData(
                src[i], mt[i], mt_infos[i].target_bpe, mt_infos[i].model_scores, retain_eos=self.retain_eos
            )
            item_labels = list(map(int, labels[i].split()))
            try:
                item_labels = map_token_labels_to_word_labels(item.target_moses, item.target_words, item_labels)
            except KeyError:
                continue
            self.data.append((item, item_labels))

    def ordered_indices(self, *args):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices

    def __len__(self):
        return len(self.data)


class CVDataset(Dataset):

    def __init__(self, shuffle, K, retain_eos=False):
        super().__init__(shuffle, retain_eos=retain_eos)
        self.K = K
        self.folds = []

    def make_folds(self):
        indices = np.random.permutation(len(self)).astype(np.int64)
        fold_size = len(self) // self.K
        for i in range(self.K + 1):
            self.folds.append(indices[fold_size * i: min(fold_size * (i + 1), len(self))])

    def get_train_folds(self, test_fold_id):
        size = len(self) - len(self.folds[test_fold_id])
        indices = np.ndarray((size,), dtype=np.int64)
        pos = 0
        for i in range(len(self.folds)):
            if i == test_fold_id:
                continue
            indices[pos:pos+len(self.folds[i])] = self.folds[i]
            pos = pos+len(self.folds[i])
        return indices

    def ordered_indices(self, fold):
        return self.folds[fold]
