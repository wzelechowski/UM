import numpy as np
from similarity.agreement_person_similarity import AgreementPersonSimilarity
from similarity.pearson_person_similarity import PearsonPersonSimilarity
from similarity.cosine_person_similarity import CosinePersonSimilarity


class ProcessingService:
    def __init__(self, person_similarities):
        self._person_similarities = person_similarities

    @classmethod
    def create_default(cls):
        default_similarities = {
            'agreement_s3': AgreementPersonSimilarity(max_diff=5.0, min_common=2, shrinkage=3.0),
            'agreement_s7': AgreementPersonSimilarity(max_diff=5.0, min_common=2, shrinkage=7.0),
            'pearson_s5': PearsonPersonSimilarity(min_common=3, shrinkage=5.0),
            'cosine_s5': CosinePersonSimilarity(min_common=2, shrinkage=5.0),
        }
        return cls(person_similarities=default_similarities)

    def build_ratings_and_similarity_matrices(self, df_train, df_task):
        user_ids = sorted(set(df_train['user_id']).union(set(df_task['user_id'])))
        movie_ids = sorted(set(df_train['movie_id']).union(set(df_task['movie_id'])))

        user_to_idx = {u_id: i for i, u_id in enumerate(user_ids)}
        movie_to_idx = {m_id: i for i, m_id in enumerate(movie_ids)}

        ratings_matrix = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=np.float32)

        for row in df_train.itertuples(index=False):
            u_idx = user_to_idx[row.user_id]
            m_idx = movie_to_idx[row.movie_id]
            ratings_matrix[u_idx, m_idx] = float(row.evaluation)

        print(f"Obliczanie macierzy podobieństwa dla {len(user_ids)} osób na bazie {len(movie_ids)} filmów...")

        sim_matrices = {
            name: sim.compute_matrix(ratings_matrix)
            for name, sim in self._person_similarities.items()
        }

        return user_to_idx, movie_to_idx, ratings_matrix, sim_matrices
