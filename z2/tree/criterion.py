from abc import ABC, abstractmethod
import numpy as np


class SplitCriterion(ABC):
    @abstractmethod
    def impurity(self, y):
        pass

    def information_gain(self, y_parent, y_left, y_right):
        n = len(y_parent)
        if n == 0 or len(y_left) == 0 or len(y_right) == 0:
            return 0.0

        parent_imp = self.impurity(y_parent)
        w_left = len(y_left) / n
        w_right = len(y_right) / n

        children_imp = w_left * self.impurity(y_left) + w_right * self.impurity(y_right)
        return parent_imp - children_imp


class GiniCriterion(SplitCriterion):
    def impurity(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(1.0 - np.sum(probs ** 2))


class EntropyCriterion(SplitCriterion):
    def impurity(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
