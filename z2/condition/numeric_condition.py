import numpy as np
import pandas as pd
from condition.condition import Condition, ConditionFactory


class NumericCondition(Condition):
    def __init__(self, feature_name, threshold):
        super().__init__(feature_name)
        self.threshold = float(threshold)

    def evaluate_mask(self, series):
        vals = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
        return np.where(np.isnan(vals), False, vals <= self.threshold)

    def is_satisfied(self, movie_row):
        val = movie_row.get(self.feature_name)
        if val is None or pd.isna(val):
            return False
        return float(val) <= self.threshold

    def to_label(self):
        return f"{self.feature_name} <= {self.threshold:.2f}"


class NumericConditionFactory(ConditionFactory):
    def __init__(self, max_thresholds=20):
        self.max_thresholds = max_thresholds

    def generate_conditions(self, feature_name, series):
        vals = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=np.float64)
        unique_vals = np.unique(vals)

        if len(unique_vals) <= 1:
            return []

        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        if len(midpoints) > self.max_thresholds:
            indices = np.linspace(0, len(midpoints) - 1, self.max_thresholds, dtype=int)
            midpoints = midpoints[indices]

        return [NumericCondition(feature_name, float(t)) for t in midpoints]
