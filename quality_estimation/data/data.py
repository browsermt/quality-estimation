import re


class InputData:

    def __init__(self, source, target_moses, target_bpe, model_scores, retain_eos=False, bpe_sep='▁'):
        """
        :param source: str
        :param target: str
        :param target_bpe_tokens: str
        :param model_scores: List[float]
        We assume target_bpe and model scores include eos token, which is ignored for now,
        but might be useful for a later version
        """
        self.bpe_sep = bpe_sep
        self.retain_eos = retain_eos
        self.source = source
        self.model_scores = model_scores
        self.target_bpe = target_bpe
        if not self.retain_eos:
            self.model_scores = self.model_scores[:-1]
            self.target_bpe = self.target_bpe.replace("</s>", "").strip()

        self.target_moses = target_moses
        self.target_words = self.remove_bpe()

    def remove_bpe(self):
        rep = re.sub("\s", "", self.target_bpe.lstrip(self.bpe_sep))
        rep = re.sub(self.bpe_sep, " ", rep)
        return rep


class Output:

    def __init__(self, sentence_score, word_scores):
        """
        :param word_scores: np.ndarray of shape (num_words,)
        :param sentence_score: float
        """
        self.sentence_score = sentence_score
        self.word_scores = word_scores
