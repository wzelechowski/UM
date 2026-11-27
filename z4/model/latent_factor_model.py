import numpy as np


class LatentFactorModel:
    def __init__(self, n_users, n_movies, n_factors=10, use_mean_normalization=True, random_seed=42):
        self.n_users = int(n_users)
        self.n_movies = int(n_movies)
        self.n_factors = int(n_factors)
        self.use_mean_normalization = bool(use_mean_normalization)
        self.rng = np.random.default_rng(random_seed)

        # X: cechy ukryte filmów (n_movies x n_factors)
        # Theta: parametry preferencji użytkowników (n_users x n_factors)
        self.X = self.rng.normal(0.0, 0.1, size=(self.n_movies, self.n_factors))
        self.Theta = self.rng.normal(0.0, 0.1, size=(self.n_users, self.n_factors))
        self.movie_means = np.zeros(self.n_movies, dtype=np.float64)
        self.global_mean = 3.0

    def reset_parameters(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self.X = rng.normal(0.0, 0.1, size=(self.n_movies, self.n_factors))
        self.Theta = rng.normal(0.0, 0.1, size=(self.n_users, self.n_factors))

    def compute_movie_means(self, u_indices, m_indices, ratings):
        if len(ratings) > 0:
            self.global_mean = float(np.mean(ratings))
        else:
            self.global_mean = 3.0

        self.movie_means = np.full(self.n_movies, self.global_mean, dtype=np.float64)
        if not self.use_mean_normalization:
            self.movie_means[:] = 0.0
            return

        sums = np.zeros(self.n_movies, dtype=np.float64)
        counts = np.zeros(self.n_movies, dtype=np.int64)
        np.add.at(sums, m_indices, ratings)
        np.add.at(counts, m_indices, 1)

        rated_mask = counts > 0
        self.movie_means[rated_mask] = sums[rated_mask] / counts[rated_mask]

    def predict_pair(self, u_idx, m_idx):
        raw_pred = float(np.dot(self.Theta[u_idx], self.X[m_idx]))
        if self.use_mean_normalization:
            raw_pred += float(self.movie_means[m_idx])
        return raw_pred

    def predict_batch(self, u_indices, m_indices):
        raw_preds = np.sum(self.Theta[u_indices] * self.X[m_indices], axis=1)
        if self.use_mean_normalization:
            raw_preds += self.movie_means[m_indices]
        return raw_preds
