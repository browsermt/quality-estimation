import unittest

import numpy as np

from quality_estimation.data.datasets import Dataset, CVDataset
from quality_estimation.data.utils import map_token_labels_to_word_labels
from quality_estimation.data.data import InputData


def make_input_data():
    item = InputData(
        source="This is an example.",
        target_moses="Es un ejemplo .",
        target_bpe="▁Es ▁un ▁ej em plo. </s>",
        model_scores=[-0.0001, -0.002, -0.5, -0.2, -0.1, -0.1],
    )
    return item


class TestUtil(unittest.TestCase):

    def test_map_token_labels_to_word_labels(self):
        item = make_input_data()
        labels = [0, 0, 1, 0]
        labels = map_token_labels_to_word_labels(item.target_moses, item.target_words, labels)
        assert len(labels) == 3


class TestDataset(unittest.TestCase):

    def test_collate_fn(self):
        dataset = Dataset(shuffle=False)
        item = make_input_data()
        item_labels = [0, 0, 1]
        dataset.data.append((item, item_labels))
        indices = np.zeros(1, dtype=np.int64)
        F, labels = dataset.collate_fn(indices)
        assert F.shape == (3, 4)
        assert labels.shape == (3,)


class TestCVDataset(unittest.TestCase):

    def test_make_folds(self):
        dataset = CVDataset(shuffle=False, K=5)
        dataset.data = [i for i in range(11)]
        dataset.make_folds()
        test_indices = dataset.folds[0]
        train_indices = dataset.get_train_folds(0)
        assert len(dataset.folds) == 6
        assert len(train_indices) == 9
        assert len(test_indices) == 2
