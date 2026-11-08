from abc import ABC, abstractmethod
import numpy as np


class TreeNode(ABC):
    def __init__(self, impurity, samples_count, class_counts=None, class_probs=None):
        self.impurity = float(impurity)
        self.samples_count = int(samples_count)
        self.class_counts = (
            class_counts if class_counts is not None else np.zeros(6, dtype=int)
        )
        self.class_probs = (
            class_probs if class_probs is not None else np.ones(6, dtype=float) / 6.0
        )

    @abstractmethod
    def predict_one(self, movie_row):
        pass

    @abstractmethod
    def predict_proba_one(self, movie_row):
        pass

    @abstractmethod
    def get_depth(self):
        pass

    @abstractmethod
    def count_nodes_and_leaves(self):
        pass


class LeafNode(TreeNode):
    def __init__(self, value, impurity, samples_count, class_counts=None, class_probs=None):
        super().__init__(impurity, samples_count, class_counts, class_probs)
        self.value = int(value)

    def predict_one(self, movie_row):
        return self.value

    def predict_proba_one(self, movie_row):
        return self.class_probs

    def get_depth(self):
        return 1

    def count_nodes_and_leaves(self):
        return 1, 1


class DecisionNode(TreeNode):
    def __init__(self, condition, left, right, impurity, samples_count, class_counts=None, class_probs=None):
        super().__init__(impurity, samples_count, class_counts, class_probs)
        self.condition = condition
        self.left = left
        self.right = right

    def predict_one(self, movie_row):
        if self.condition.is_satisfied(movie_row):
            return self.left.predict_one(movie_row)
        return self.right.predict_one(movie_row)

    def predict_proba_one(self, movie_row):
        if self.condition.is_satisfied(movie_row):
            return self.left.predict_proba_one(movie_row)
        return self.right.predict_proba_one(movie_row)

    def get_depth(self):
        return 1 + max(self.left.get_depth(), self.right.get_depth())

    def count_nodes_and_leaves(self):
        l_nodes, l_leaves = self.left.count_nodes_and_leaves()
        r_nodes, r_leaves = self.right.count_nodes_and_leaves()
        return 1 + l_nodes + r_nodes, l_leaves + r_leaves
