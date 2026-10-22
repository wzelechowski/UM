from abc import ABC, abstractmethod

class Similarity(ABC):
    @abstractmethod
    def compute_matrix(self, series):
        pass