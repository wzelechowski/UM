import numpy as np


class LeaveOneOutValidator:
    def __init__(self, k_candidates=None, aggregation_candidates=None):
        self.k_candidates = k_candidates or [5, 10, 15, 25, 40]
        self.aggregation_candidates = aggregation_candidates or ['mean_centered', 'weighted', 'uniform']

    def select_best_user_config(self, u_idx, ratings_matrix, user_means, sim_matrices):
        u_ratings = ratings_matrix[u_idx]
        rated_movie_indices = np.flatnonzero(~np.isnan(u_ratings))
        u_mean = float(user_means[u_idx])

        best_mse = float('inf')
        best_config = {
            'sim_name': next(iter(sim_matrices.keys())),
            'k': 15,
            'aggregation': 'mean_centered',
        }

        if len(rated_movie_indices) < 3:
            return best_config

        for sim_name, sim_mat in sim_matrices.items():
            u_sims = sim_mat[u_idx]

            for k in self.k_candidates:
                for agg in self.aggregation_candidates:
                    squared_errors = []

                    for m_idx in rated_movie_indices:
                        true_val = float(u_ratings[m_idx])
                        pred_val = self.predict_single(
                            u_idx=u_idx,
                            m_idx=m_idx,
                            ratings_matrix=ratings_matrix,
                            user_means=user_means,
                            u_sims=u_sims,
                            k=k,
                            aggregation=agg,
                            fallback_rating=u_mean,
                        )
                        squared_errors.append((true_val - pred_val) ** 2)

                    mse = float(np.mean(squared_errors))
                    if mse < best_mse:
                        best_mse = mse
                        best_config = {
                            'sim_name': sim_name,
                            'k': k,
                            'aggregation': agg,
                        }

        return best_config

    @staticmethod
    def predict_single(u_idx, m_idx, ratings_matrix, user_means, u_sims, k, aggregation, fallback_rating):
        col_ratings = ratings_matrix[:, m_idx]
        valid_mask = ~np.isnan(col_ratings)
        valid_mask[u_idx] = False

        candidate_users = np.flatnonzero(valid_mask)
        if len(candidate_users) == 0:
            return fallback_rating

        candidate_sims = u_sims[candidate_users]
        positive_mask = candidate_sims > 1e-6

        if not np.any(positive_mask):
            return fallback_rating

        candidate_users = candidate_users[positive_mask]
        candidate_sims = candidate_sims[positive_mask]

        if len(candidate_users) > k:
            top_k_Order = np.argsort(candidate_sims)[::-1][:k]
            neighbor_users = candidate_users[top_k_Order]
            neighbor_sims = candidate_sims[top_k_Order]
        else:
            neighbor_users = candidate_users
            neighbor_sims = candidate_sims

        neighbor_ratings = col_ratings[neighbor_users]
        weights_sum = float(np.sum(neighbor_sims))

        if weights_sum <= 1e-6:
            return fallback_rating

        if aggregation == 'uniform':
            pred = float(np.mean(neighbor_ratings))
        elif aggregation == 'weighted':
            pred = float(np.dot(neighbor_sims, neighbor_ratings) / weights_sum)
        else:
            neighbor_means = user_means[neighbor_users]
            centered_ratings = neighbor_ratings - neighbor_means
            pred = float(user_means[u_idx] + (np.dot(neighbor_sims, centered_ratings) / weights_sum))

        return float(np.clip(pred, 0.0, 5.0))
