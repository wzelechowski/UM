import numpy as np
from optimizer.cf_optimizer import CFOptimizer


class AlternatingLeastSquaresOptimizer(CFOptimizer):
    def __init__(self, reg_lambda=0.5, n_epochs=25):
        super().__init__(reg_lambda=reg_lambda, n_epochs=n_epochs)

    def fit(self, model, u_indices, m_indices, ratings):
        model.compute_movie_means(u_indices, m_indices, ratings)
        target_ratings = ratings - model.movie_means[m_indices]
        k = model.n_factors
        eye_k = self.reg_lambda * np.eye(k, dtype=np.float64)

        user_movies = [[] for _ in range(model.n_users)]
        user_targets = [[] for _ in range(model.n_users)]
        movie_users = [[] for _ in range(model.n_movies)]
        movie_targets = [[] for _ in range(model.n_movies)]

        for u, m, y in zip(u_indices, m_indices, target_ratings):
            user_movies[u].append(m)
            user_targets[u].append(y)
            movie_users[m].append(u)
            movie_targets[m].append(y)

        user_movies = [np.array(lst, dtype=int) for lst in user_movies]
        user_targets = [np.array(lst, dtype=np.float64) for lst in user_targets]
        movie_users = [np.array(lst, dtype=int) for lst in movie_users]
        movie_targets = [np.array(lst, dtype=np.float64) for lst in movie_targets]

        # Naprzemienna aktualizacja Theta (przy zamrożonym X) oraz X (przy zamrożonym Theta) - Hint 1
        for _ in range(self.n_epochs):
            # 1. Aktualizacja parametrów użytkowników Theta przy stałych cechach filmów X
            for u in range(model.n_users):
                m_idx = user_movies[u]
                if len(m_idx) == 0:
                    continue
                X_u = model.X[m_idx]
                y_u = user_targets[u]
                A = X_u.T @ X_u + eye_k
                b = X_u.T @ y_u
                model.Theta[u] = np.linalg.solve(A, b)

            # 2. Aktualizacja cech ukrytych filmów X przy stałych parametrach użytkowników Theta
            for m in range(model.n_movies):
                u_idx = movie_users[m]
                if len(u_idx) == 0:
                    continue
                Theta_m = model.Theta[u_idx]
                y_m = movie_targets[m]
                A = Theta_m.T @ Theta_m + eye_k
                b = Theta_m.T @ y_m
                model.X[m] = np.linalg.solve(A, b)

        return model
