from abc import ABC, abstractmethod


class CFOptimizer(ABC):
    def __init__(self, reg_lambda=0.1, n_epochs=100):
        self.reg_lambda = float(reg_lambda)
        self.n_epochs = int(n_epochs)

    @abstractmethod
    def fit(self, model, u_indices, m_indices, ratings):
        pass
