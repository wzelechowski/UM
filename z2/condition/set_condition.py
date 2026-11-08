from collections import Counter
import numpy as np
from condition.condition import Condition, ConditionFactory


class SetCondition(Condition):
    def __init__(self, feature_name, element):
        super().__init__(feature_name)
        self.element = element

    def evaluate_mask(self, series):
        return np.array([
            isinstance(x, (list, set, tuple)) and self.element in x
            for x in series
        ], dtype=bool)

    def is_satisfied(self, movie_row):
        items = movie_row.get(self.feature_name)
        if not isinstance(items, (list, set, tuple)):
            return False
        return self.element in items

    def to_label(self):
        return f"'{self.element}' in {self.feature_name}"


class SetConditionFactory(ConditionFactory):
    def __init__(self, min_support=2, max_candidates=25):
        self.min_support = min_support
        self.max_candidates = max_candidates

    def generate_conditions(self, feature_name, series):
        n = len(series)
        counts = Counter()

        for items in series:
            if isinstance(items, (list, set, tuple)):
                for elem in set(items):
                    if elem:
                        counts[elem] += 1

        candidates = [
            elem for elem, count in counts.most_common(self.max_candidates)
            if self.min_support <= count <= (n - self.min_support)
        ]

        return [SetCondition(feature_name, elem) for elem in candidates]
