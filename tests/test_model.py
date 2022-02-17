import numpy as np
import unittest

from quality_estimation.data.data import InputData
from quality_estimation.model import Model


class TestModel(unittest.TestCase):

    def test_compute_quality_scores(self):
        coefficients = np.asarray([0.99, 0.9, -0.2, 0.5])
        coefficients = np.expand_dims(coefficients, axis=0)
        intercept = np.asarray([-0.3])
        means = np.asarray([-0.1, -0.77, 5., -0.5])
        stds = np.asarray([0.2, 0.3, 2.5, 0.1])

        qe_model = Model(coefficients, intercept, means, stds)
        input_data = InputData(
            source='This is an example.',
            target_moses='Es un ejemplo.',
            target_bpe='▁Es ▁un ▁ej em plo. </s>',
            model_scores=[-0.0001, -0.002, -0.5, -0.2, -0.1, -0.001]
        )
        output = qe_model.compute_quality_scores(input_data)
        assert len(output.word_scores) == len(input_data.target_moses.split())
        assert output.sentence_score == 0.


if __name__ == "__main__":
    unittest.main()
