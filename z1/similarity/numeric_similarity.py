import numpy as np
from similarity.similarity import Similarity


class NumericSimilarity(Similarity):
    def __init__(self, default_max_diff=1.0):
        self.default_max_diff = default_max_diff

    def compute_matrix(self, series):
        vals = series.to_numpy(dtype=np.float32)
        nan_mask = np.isnan(vals)

        valid_vals = vals[~nan_mask]
        if len(valid_vals) > 1:
            max_diff = float(np.max(valid_vals) - np.min(valid_vals))
        else:
            max_diff = 0.0

        if max_diff <= 0:
            max_diff = self.default_max_diff

        diff_matrix = np.abs(vals[:, None] - vals[None, :])
        sim_matrix = np.clip(1.0 - (diff_matrix / max_diff), 0.0, 1.0)

        if nan_mask.any():
            sim_matrix[nan_mask, :] = 0.0
            sim_matrix[:, nan_mask] = 0.0

        np.fill_diagonal(sim_matrix, 1.0)
        return sim_matrix.astype(np.float32)