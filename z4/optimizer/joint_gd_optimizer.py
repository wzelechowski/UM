import numpy as np
from optimizer.cf_optimizer import CFOptimizer


class JointGradientDescentOptimizer(CFOptimizer):
    def __init__(self, lr=0.01, reg_lambda=0.1, n_epochs=150):
        super().__init__(reg_lambda=reg_lambda, n_epochs=n_epochs)
        self.lr = float(lr)

    def fit(self, model, u_indices, m_indices, ratings):
        model.compute_movie_means(u_indices, m_indices, ratings)
        target_ratings = ratings - model.movie_means[m_indices]
        n_samples = len(target_ratings)
        rng = np.random.default_rng(42)

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            u_perm = u_indices[perm]
            m_perm = m_indices[perm]
            y_perm = target_ratings[perm]

            # Wektorowa aktualizacja obu macierzy jednocześnie (Hint 1: updated together)
            for start in range(0, n_samples, 256):
                end = start + 256
                ub = u_perm[start:end]
                mb = m_perm[start:end]
                yb = y_perm[start:end]

                preds = np.sum(model.Theta[ub] * model.X[mb], axis=1)
                errors = preds - yb

                theta_u = model.Theta[ub].copy()
                x_m = model.X[mb].copy()

                grad_theta = errors[:, None] * x_m + self.reg_lambda * theta_u
                grad_x = errors[:, None] * theta_u + self.reg_lambda * x_m

                np.add.at(model.Theta, ub, -self.lr * grad_theta)
                np.add.at(model.X, mb, -self.lr * grad_x)

        return model
