from abc import ABC, abstractmethod


class PersonSimilarity(ABC):
    @abstractmethod
    def compute_matrix(self, ratings_matrix):
        pass
