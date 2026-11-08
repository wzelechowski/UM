import random
import numpy as np
from tree.node import LeafNode, DecisionNode
from tree.criterion import GiniCriterion


class BinaryDecisionTree:
    def __init__(
        self,
        criterion=None,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=2,
        min_impurity_decrease=0.001,
        max_features_ratio=None,
        n_classes=6,
        random_state=42,
    ):
        self.criterion = criterion or GiniCriterion()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features_ratio = max_features_ratio
        self.n_classes = int(n_classes)
        self.rng = random.Random(random_state)
        self.root = None
        self.user_prior = np.ones(self.n_classes, dtype=float) / self.n_classes
        self.user_mean_rating = 3.0

    def fit(self, df_X, y, condition_factories):
        y_arr = np.clip(np.asarray(y, dtype=int), 0, self.n_classes - 1)
        df_reset = df_X.reset_index(drop=True)

        if len(y_arr) > 0:
            total_counts = np.bincount(y_arr, minlength=self.n_classes).astype(float)
            self.user_prior = total_counts / float(len(y_arr))
            self.user_mean_rating = float(np.mean(y_arr))

        self.root = self._induce_node(df_reset, y_arr, condition_factories, depth=0)
        return self

    def predict_row(self, movie_row):
        if self.root is None:
            return int(round(self.user_mean_rating))
        return self.root.predict_one(movie_row)

    def predict_proba_row(self, movie_row):
        if self.root is None:
            return self.user_prior
        return self.root.predict_proba_one(movie_row)

    def predict(self, df_X):
        records = df_X.to_dict('records')
        return np.array([self.predict_row(row) for row in records], dtype=int)

    def predict_proba(self, df_X):
        records = df_X.to_dict('records')
        return np.array([self.predict_proba_row(row) for row in records], dtype=float)

    def _induce_node(self, df_X, y, condition_factories, depth):
        n_samples = len(y)
        class_counts = np.bincount(y, minlength=self.n_classes).astype(int)
        class_probs = class_counts / max(1.0, float(n_samples))
        impurity = self.criterion.impurity(y)
        predicted_class = self._resolve_leaf_class(class_counts)

        if self._should_stop(depth, n_samples, impurity):
            return LeafNode(predicted_class, impurity, n_samples, class_counts, class_probs)

        best_split = self._find_best_split(df_X, y, condition_factories)
        if best_split is None:
            return LeafNode(predicted_class, impurity, n_samples, class_counts, class_probs)

        best_condition, left_mask, right_mask = best_split

        left_node = self._induce_node(
            df_X[left_mask].reset_index(drop=True),
            y[left_mask],
            condition_factories,
            depth + 1,
        )
        right_node = self._induce_node(
            df_X[right_mask].reset_index(drop=True),
            y[right_mask],
            condition_factories,
            depth + 1,
        )

        return DecisionNode(
            best_condition, left_node, right_node, impurity, n_samples, class_counts, class_probs
        )

    def _should_stop(self, depth, n_samples, impurity):
        if impurity <= 1e-7:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self.min_samples_split or n_samples < 2 * self.min_samples_leaf:
            return True
        return False

    def _find_best_split(self, df_X, y, condition_factories):
        available_features = [f for f in condition_factories.keys() if f in df_X.columns]
        selected_features = self._subsample_features(available_features)

        best_gain = max(1e-7, self.min_impurity_decrease)
        best_split = None

        for feat in selected_features:
            factory = condition_factories[feat]
            candidates = factory.generate_conditions(feat, df_X[feat])

            for condition in candidates:
                left_mask = condition.evaluate_mask(df_X[feat])
                right_mask = ~left_mask

                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = self.criterion.information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_split = (condition, left_mask, right_mask)

        return best_split

    def _subsample_features(self, features):
        if not features:
            return []
        if self.max_features_ratio is None or self.max_features_ratio == 'all':
            return features
        if self.max_features_ratio == 'sqrt':
            k = max(1, int(np.sqrt(len(features))))
        elif self.max_features_ratio == 'log2':
            k = max(1, int(np.log2(len(features))))
        elif isinstance(self.max_features_ratio, float) and 0.0 < self.max_features_ratio < 1.0:
            k = max(1, int(round(len(features) * self.max_features_ratio)))
        else:
            return features
        return self.rng.sample(features, min(len(features), k))

    def _resolve_leaf_class(self, class_counts):
        max_cnt = np.max(class_counts)
        candidates = np.flatnonzero(class_counts == max_cnt)
        if len(candidates) == 1:
            return int(candidates[0])

        priors = self.user_prior[candidates]
        max_prior = np.max(priors)
        candidates = candidates[np.isclose(priors, max_prior)]
        if len(candidates) == 1:
            return int(candidates[0])

        dists = np.abs(candidates - self.user_mean_rating)
        min_dist = np.min(dists)
        candidates = candidates[np.isclose(dists, min_dist)]
        if len(candidates) == 1:
            return int(candidates[0])

        dists_global = np.abs(candidates - 3.0)
        return int(candidates[np.argmin(dists_global)])
