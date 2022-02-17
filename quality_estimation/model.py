import numpy as np

from .data.data import Output
from .data.utils import map_words_to_bpe_tokens


class Model:

    def __init__(self, coefficients, intercept, feature_means, features_stds):
        """
        :param coefficients: np.ndarray of shape (1, num_features)
        :param intercept: np.ndarray of shape (1,)
        :param feature_means: np.ndarray of shape (num_features,) for feature scaling
        :param feature_stds: np.ndarray of shape (num_features,) for feature scaling
        """
        self.coefficients = coefficients
        self.intercept = intercept
        self.classes = np.asarray([0, 1])
        self.feature_means = feature_means
        self.feature_stds = features_stds

    def compute_quality_scores(self, input_data):
        """
        :param input_data: object of efficient_qe.data.data.InputData class
        :return: object of efficient_qe.data.data.Output class
        """
        X = self._extract_features(input_data)
        word_labels = self._predict_word_level(X)
        sent_score = 1. - sum(word_labels)/len(word_labels)
        return Output(sent_score, word_labels)

    def _predict_word_level(self, X):
        scores = np.dot(X, self.coefficients.T) + self.intercept
        scores = scores.ravel()
        indices = (scores > 0).astype(np.int)
        return self.classes[indices]

    def _extract_features(self, input_data):
        mapping = map_words_to_bpe_tokens(input_data.target_bpe)
        X = np.ndarray((len(mapping), 4))
        # TODO: features are hard-coded for now
        # We will need to change that if we implement a more advanced version (e.g. include LM-based features)
        for word_index, bpe_indices in enumerate(mapping):
            bpe_scores = [input_data.model_scores[bpe_idx] for bpe_idx in bpe_indices]
            X[word_index, 0] = sum(bpe_scores)/len(bpe_scores)
            X[word_index, 1] = min(bpe_scores)
            X[word_index, 2] = len(bpe_scores)
            X[word_index, 3] = sum(input_data.model_scores)/len(input_data.model_scores)
        X = (X - self.feature_means) / self.feature_stds  # scale features
        return X
