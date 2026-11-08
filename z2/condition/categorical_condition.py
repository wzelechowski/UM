from collections import Counter
import numpy as np
import pandas as pd
from condition.condition import Condition, ConditionFactory


class CategoricalCondition(Condition):
    def __init__(self, feature_name, category):
        super().__init__(feature_name)
        self.category = str(category)

    def evaluate_mask(self, series):
        vals = series.fillna('').astype(str).str.strip()
        return (vals == self.category).to_numpy(dtype=bool)

    def is_satisfied(self, movie_row):
        val = movie_row.get(self.feature_name)
        if val is None or pd.isna(val):
            return False
        return str(val).strip() == self.category

    def to_label(self):
        return f"{self.feature_name} == '{self.category}'"


class CategoricalConditionFactory(ConditionFactory):
    def __init__(self, min_support=2, max_candidates=20):
        self.min_support = min_support
        self.max_candidates = max_candidates

    def generate_conditions(self, feature_name, series):
        n = len(series)
        cleaned = [
            str(x).strip()
            for x in series
            if x is not None and not pd.isna(x) and str(x).strip() != ''
        ]
        counts = Counter(cleaned)

        candidates = [
            val for val, count in counts.most_common(self.max_candidates)
            if self.min_support <= count <= (n - self.min_support)
        ]

        return [CategoricalCondition(feature_name, val) for val in candidates]
