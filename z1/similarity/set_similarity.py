import numpy as np
from similarity.similarity import Similarity

class SetSimilarity(Similarity):

    def compute_matrix(self, series):
        items = [set(x) if isinstance(x, (list, set)) and x else set() for x in series]
        n = len(items)
        sim_matrix = np.eye(n, dtype=np.float32)

        for i in range(n):
            set1 = items[i]
            if not set1:
                continue
            for j in range(i + 1, n):
                set2 = items[j]
                if not set2:
                    continue
                union = len(set1 | set2)
                if union > 0:
                    val = len(set1 & set2) / union
                    sim_matrix[i, j] = val
                    sim_matrix[j, i] = val

        return sim_matrix