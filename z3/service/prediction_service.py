import numpy as np
from service.leave_one_out_validator import LeaveOneOutValidator


class PredictionService:
    def __init__(self, validator):
        self._validator = validator

    @classmethod
    def create_default(cls):
        return cls(validator=LeaveOneOutValidator())

    def predict_for_all_users(self, df_train, df_task, user_to_idx, movie_to_idx, ratings_matrix, sim_matrices):
        df_submission = df_task.copy()
        n_users = ratings_matrix.shape[0]

        user_means = np.zeros(n_users, dtype=np.float32)
        for u in range(n_users):
            u_vals = ratings_matrix[u, ~np.isnan(ratings_matrix[u])]
            user_means[u] = float(np.mean(u_vals)) if len(u_vals) > 0 else 3.0

        unique_users = df_submission['user_id'].unique()
        print(f"Rozpoczynam dobór parametrów (CV) i predykcję User-Based CF dla {len(unique_users)} osób...")

        for idx, user_id in enumerate(unique_users, 1):
            user_task_mask = df_submission['user_id'] == user_id
            user_task_rows = df_submission[user_task_mask]

            if user_id not in user_to_idx:
                df_submission.loc[user_task_mask, 'evaluation'] = 3
                continue

            u_idx = user_to_idx[user_id]
            fallback_rating = float(user_means[u_idx])

            best_config = self._validator.select_best_user_config(
                u_idx, ratings_matrix, user_means, sim_matrices
            )

            best_sim_mat = sim_matrices[best_config['sim_name']]
            u_sims = best_sim_mat[u_idx]
            best_k = best_config['k']
            best_agg = best_config['aggregation']

            predictions = []
            for movie_id in user_task_rows['movie_id']:
                if movie_id in movie_to_idx:
                    m_idx = movie_to_idx[movie_id]
                    pred = self._validator.predict_single(
                        u_idx=u_idx,
                        m_idx=m_idx,
                        ratings_matrix=ratings_matrix,
                        user_means=user_means,
                        u_sims=u_sims,
                        k=best_k,
                        aggregation=best_agg,
                        fallback_rating=fallback_rating,
                    )
                    predictions.append(int(np.clip(round(pred), 0, 5)))
                else:
                    predictions.append(int(np.clip(round(fallback_rating), 0, 5)))

            df_submission.loc[user_task_mask, 'evaluation'] = predictions

            if idx % 50 == 0 or idx == len(unique_users):
                print(f"Przetworzono {idx}/{len(unique_users)} osób...")

        df_submission['evaluation'] = df_submission['evaluation'].astype(int)
        return df_submission
