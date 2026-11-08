import re
from collections import Counter
import numpy as np
from condition.condition import Condition, ConditionFactory

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "his", "her",
    "they", "their", "who", "when", "after", "into", "about", "are", "was",
    "movie", "film", "part", "story"
}


class TextWordCondition(Condition):
    def __init__(self, feature_name, word):
        super().__init__(feature_name)
        self.word = str(word).lower()

    def evaluate_mask(self, series):
        texts = series.fillna('').astype(str).str.lower()
        return np.array([
            self.word in re.findall(r"\b[a-z]{3,}\b", t)
            for t in texts
        ], dtype=bool)

    def is_satisfied(self, movie_row):
        text = str(movie_row.get(self.feature_name) or "").lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        return self.word in words

    def to_label(self):
        return f"'{self.word}' in {self.feature_name}"


class TextConditionFactory(ConditionFactory):
    def __init__(self, min_support=2, max_candidates=15):
        self.min_support = min_support
        self.max_candidates = max_candidates

    def generate_conditions(self, feature_name, series):
        n = len(series)
        counts = Counter()

        for text in series.fillna('').astype(str).str.lower():
            words = {w for w in re.findall(r"\b[a-z]{3,}\b", text) if w not in STOPWORDS}
            for w in words:
                counts[w] += 1

        candidates = [
            w for w, count in counts.most_common(self.max_candidates)
            if self.min_support <= count <= (n - self.min_support)
        ]

        return [TextWordCondition(feature_name, w) for w in candidates]
