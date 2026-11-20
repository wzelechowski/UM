import numpy as np
from similarity.person_similarity import PersonSimilarity


class AgreementPersonSimilarity(PersonSimilarity):
    def __init__(self, max_diff=5.0, min_common=2, shrinkage=5.0):
        self.max_diff = float(max_diff)
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

                diffs = np.abs(u_ratings[common_mask] - ratings_matrix[v, common_mask])
                mean_diff = float(np.mean(diffs))
                raw_sim = max(0.0, 1.0 - (mean_diff / self.max_diff))

                confidence = n_common / (n_common + self.shrinkage)
                val = float(raw_sim * confidence)

                sim_matrix[u, v] = val
                sim_matrix[v, u] = val

        return sim_matrix
