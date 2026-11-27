import numpy as np
from model.latent_factor_model import LatentFactorModel
from service.cross_validation_validator import CrossValidationValidator


class PredictionService:
    def __init__(self, validator):
        self._validator = validator

    @classmethod
    def create_default(cls):
        return cls(validator=CrossValidationValidator(n_splits=4))

    def predict_for_all_users(self, df_train, df_task, user_to_idx, movie_to_idx, u_indices, m_indices, ratings):
        n_users = len(user_to_idx)
        n_movies = len(movie_to_idx)

        best_cfg = self._validator.select_best_config(
            n_users, n_movies, u_indices, m_indices, ratings
        )

        final_model = LatentFactorModel(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=best_cfg['n_factors'],
            use_mean_normalization=best_cfg['mean_norm'],
            random_seed=42,
        )
        optimizer = self._validator.build_optimizer(best_cfg)
        optimizer.fit(final_model, u_indices, m_indices, ratings)

        df_submission = df_task.copy()
        task_u = np.array([user_to_idx[u] for u in df_submission['user_id']], dtype=int)
        task_m = np.array([movie_to_idx[m] for m in df_submission['movie_id']], dtype=int)

        raw_preds = final_model.predict_batch(task_u, task_m)
        int_preds = np.clip(np.rint(raw_preds), 0, 5).astype(int)

        df_submission['evaluation'] = int_preds
        return df_submission
