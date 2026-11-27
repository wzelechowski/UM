import numpy as np


class ProcessingService:
    def prepare_interaction_data(self, df_train, df_task):
        user_ids = sorted(set(df_train['user_id']).union(set(df_task['user_id'])))
        movie_ids = sorted(set(df_train['movie_id']).union(set(df_task['movie_id'])))

        user_to_idx = {u_id: i for i, u_id in enumerate(user_ids)}
        movie_to_idx = {m_id: i for i, m_id in enumerate(movie_ids)}

        u_indices = np.array([user_to_idx[u] for u in df_train['user_id']], dtype=int)
        m_indices = np.array([movie_to_idx[m] for m in df_train['movie_id']], dtype=int)
        ratings = df_train['evaluation'].to_numpy(dtype=np.float64)

        print(
            f"Przygotowano {len(ratings)} ocen dla {len(user_ids)} użytkowników "
            f"i {len(movie_ids)} filmów..."
        )
        return user_to_idx, movie_to_idx, u_indices, m_indices, ratings
