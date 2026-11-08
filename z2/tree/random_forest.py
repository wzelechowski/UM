import random
import numpy as np
from tree.binary_decision_tree import BinaryDecisionTree
from tree.criterion import GiniCriterion


class RandomForest:
    def __init__(
        self,
        n_estimators=35,
        criterion=None,
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        min_impurity_decrease=0.0,
        max_features_ratio='sqrt',
        voting='soft',
        n_classes=6,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion or GiniCriterion()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features_ratio = max_features_ratio
        self.voting = voting
        self.n_classes = int(n_classes)
        self.random_state = random_state
        self.trees = []
        self.user_prior = np.ones(self.n_classes, dtype=float) / self.n_classes
        self.user_mean_rating = 3.0

    def fit(self, df_X, y, condition_factories):
        self.trees = []
        n_samples = len(df_X)
        y_arr = np.clip(np.asarray(y, dtype=int), 0, self.n_classes - 1)
        rng = random.Random(self.random_state)

        if n_samples > 0:
            counts = np.bincount(y_arr, minlength=self.n_classes).astype(float)
            self.user_prior = counts / float(n_samples)
            self.user_mean_rating = float(np.mean(y_arr))

        for i in range(self.n_estimators):
            boot_indices = [rng.randrange(n_samples) for _ in range(n_samples)]
            df_boot = df_X.iloc[boot_indices].reset_index(drop=True)
            y_boot = y_arr[boot_indices]

            tree = BinaryDecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features_ratio=self.max_features_ratio,
                n_classes=self.n_classes,
                random_state=self.random_state + i,
            )
            tree.fit(df_boot, y_boot, condition_factories)
            tree.user_prior = self.user_prior.copy()
            tree.user_mean_rating = self.user_mean_rating
            self.trees.append(tree)

        return self

    def _resolve_tie(self, scores_vec):
        max_val = np.max(scores_vec)
        candidates = np.flatnonzero(np.isclose(scores_vec, max_val, atol=1e-12))
        if len(candidates) == 1:
            return int(candidates[0])

        priors = self.user_prior[candidates]
        max_prior = np.max(priors)
        candidates = candidates[np.isclose(priors, max_prior, atol=1e-12)]
        if len(candidates) == 1:
            return int(candidates[0])

        dists = np.abs(candidates - self.user_mean_rating)
        min_dist = np.min(dists)
        candidates = candidates[np.isclose(dists, min_dist, atol=1e-12)]
        return int(candidates[0])

    def predict_row(self, movie_row):
        if not self.trees:
            return int(round(self.user_mean_rating))

        if self.voting == 'soft':
            avg_probs = np.mean([t.predict_proba_row(movie_row) for t in self.trees], axis=0)
            return self._resolve_tie(avg_probs)
        else:
            votes = np.zeros(self.n_classes, dtype=float)
            for t in self.trees:
                votes[t.predict_row(movie_row)] += 1.0
            return self._resolve_tie(votes)

    def predict(self, df_X):
        records = df_X.to_dict('records')
        return np.array([self.predict_row(row) for row in records], dtype=int)
