import numpy as np
from .feature_selector import FeatureSelector


class LeaveOneOutMseSelector(FeatureSelector):
    def rank_features(self, u_train_idx, y_train, matrices):
        scores = {}
        for comp, mat in matrices.items():
            sub_sim = mat[np.ix_(u_train_idx, u_train_idx)].copy()
            np.fill_diagonal(sub_sim, 0.0)

            weights_sum = sub_sim.sum(axis=1)
            mask = weights_sum > 0
            if mask.sum() < 3:
                scores[comp] = -999.0
                continue

            pred = (sub_sim[mask] @ y_train) / weights_sum[mask]
            mse = np.mean((y_train[mask] - pred) ** 2)
            scores[comp] = -mse

        return sorted(scores.keys(), key=lambda c: scores[c], reverse=True)