import numpy as np
from similarity.person_similarity import PersonSimilarity


class PearsonPersonSimilarity(PersonSimilarity):
    def __init__(self, min_common=3, shrinkage=5.0):
        self.min_common = int(min_common)
        self.shrinkage = float(shrinkage)

    def compute_matrix(self, ratings_matrix):
        n_users = ratings_matrix.shape[0]
        sim_matrix = np.zeros((n_users, n_users), dtype=np.float32)
        seen_mask = ~np.isnan(ratings_matrix)
        user_means = np.nanmean(ratings_matrix, axis=1)

        for u in range(n_users):
            sim_matrix[u, u] = 1.0
            u_mask = seen_mask[u]
            u_centered = ratings_matrix[u] - user_means[u]

            for v in range(u + 1, n_users):
                common_mask = u_mask & seen_mask[v]
                n_common = int(np.sum(common_mask))

                if n_common < self.min_common:
                    continue

                du = u_centered[common_mask]
                dv = (ratings_matrix[v] - user_means[v])[common_mask]

                norm_u = float(np.linalg.norm(du))
                norm_v = float(np.linalg.norm(dv))

                if norm_u > 1e-6 and norm_v > 1e-6:
                    corr = float(np.dot(du, dv) / (norm_u * norm_v))
                    raw_sim = max(0.0, corr)
                else:
                    mean_diff = float(np.mean(np.abs(ratings_matrix[u, common_mask] - ratings_matrix[v, common_mask])))
                    raw_sim = max(0.0, 1.0 - (mean_diff / 5.0))

                confidence = n_common / (n_common + self.shrinkage)
                val = float(raw_sim * confidence)

                sim_matrix[u, v] = val
                sim_matrix[v, u] = val

        return sim_matrix
