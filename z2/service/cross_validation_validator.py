import numpy as np
from tree.binary_decision_tree import BinaryDecisionTree
from tree.random_forest import RandomForest
from tree.criterion import GiniCriterion


class CrossValidationValidator:
    def __init__(self, cv_splits=3, criterion=None):
        self.cv_splits = cv_splits
        self.criterion = criterion or GiniCriterion()

    def _stratified_folds(self, y, random_state=42):
        rng = np.random.default_rng(random_state)
        n_samples = len(y)
        k = min(self.cv_splits, n_samples)
        folds = [[] for _ in range(k)]

        current_fold = 0
        for c in np.unique(y):
            c_idx = np.flatnonzero(y == c).copy()
            rng.shuffle(c_idx)
            for idx in c_idx:
                folds[current_fold % k].append(int(idx))
                current_fold += 1

        all_indices = np.arange(n_samples)
        splits = []
        for fold_indices in folds:
            val_idx = np.array(sorted(fold_indices), dtype=int)
            if len(val_idx) == 0 or len(val_idx) == n_samples:
                continue
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            splits.append((all_indices[train_mask], val_idx))
        return splits

    def _score_predictions(self, y_true, y_pred):
        accuracy = float(np.mean(y_true == y_pred))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return accuracy - 0.15 * mae

    def select_best_tree_params(self, df_X, y, condition_factories, candidate_configs):
        splits = self._stratified_folds(y)
        if not splits:
            return candidate_configs[0]

        best_score = -float('inf')
        best_params = candidate_configs[0]

        for params in candidate_configs:
            oof_preds = np.zeros_like(y)
            for train_idx, val_idx in splits:
                df_tr = df_X.iloc[train_idx].reset_index(drop=True)
                y_tr = y[train_idx]
                df_val = df_X.iloc[val_idx].reset_index(drop=True)

                tree = BinaryDecisionTree(criterion=self.criterion, **params)
                tree.fit(df_tr, y_tr, condition_factories)
                oof_preds[val_idx] = tree.predict(df_val)

            score = self._score_predictions(y, oof_preds)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params

    def select_best_forest_params(self, df_X, y, condition_factories, candidate_configs):
        splits = self._stratified_folds(y)
        if not splits:
            return candidate_configs[0]

        best_score = -float('inf')
        best_params = candidate_configs[0]

        for params in candidate_configs:
            oof_preds = np.zeros_like(y)
            for train_idx, val_idx in splits:
                df_tr = df_X.iloc[train_idx].reset_index(drop=True)
                y_tr = y[train_idx]
                df_val = df_X.iloc[val_idx].reset_index(drop=True)

                forest = RandomForest(criterion=self.criterion, **params)
                forest.fit(df_tr, y_tr, condition_factories)
                oof_preds[val_idx] = forest.predict(df_val)

            score = self._score_predictions(y, oof_preds)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params
