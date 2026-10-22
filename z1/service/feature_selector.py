from abc import ABC, abstractmethod

class FeatureSelector(ABC):
    @abstractmethod
    def rank_features(self, u_train_idx, y_train, matrices):
        pass