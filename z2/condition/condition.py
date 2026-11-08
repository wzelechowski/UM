from abc import ABC, abstractmethod


class Condition(ABC):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    @abstractmethod
    def evaluate_mask(self, series):
        pass

    @abstractmethod
    def is_satisfied(self, movie_row):
        pass

    @abstractmethod
    def to_label(self):
        pass


class ConditionFactory(ABC):
    @abstractmethod
    def generate_conditions(self, feature_name, series):
        pass
