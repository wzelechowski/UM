import numpy as np
from similarity.person_similarity import PersonSimilarity


class CosinePersonSimilarity(PersonSimilarity):
    def __init__(self, min_common=2, shrinkage=5.0):
        self.min_common = int(min_common)
        self.shrinkage = float(shrinkage)

    def compute_matrix(self, ratings_matrix):
        n_users = ratings_matrix.shape[0]
        sim_matrix = np.zeros((n_users, n_users), dtype=np.float32)
        seen_mask = ~np.isnan(ratings_matrix)

        for u in range(n_users):
            sim_matrix[u, u] = 1.0
            u_mask = seen_mask[u]
            u_ratings = ratings_matrix[u]

            for v in range(u + 1, n_users):
                common_mask = u_mask & seen_mask[v]
                n_common = int(np.sum(common_mask))

                if n_common < self.min_common:
                    continue

                ru = u_ratings[common_mask]
                rv = ratings_matrix[v, common_mask]

                norm_u = float(np.linalg.norm(ru))
                norm_v = float(np.linalg.norm(rv))

                if norm_u > 1e-6 and norm_v > 1e-6:
                    raw_sim = float(np.dot(ru, rv) / (norm_u * norm_v))
                else:
                    raw_sim = 0.0

                confidence = n_common / (n_common + self.shrinkage)
                val = float(max(0.0, raw_sim) * confidence)

                sim_matrix[u, v] = val
                sim_matrix[v, u] = val

        return sim_matrix
