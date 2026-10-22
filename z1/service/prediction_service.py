import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from .leave_one_out_mse_selector import LeaveOneOutMseSelector


class PredictionService:
    def __init__(
        self,
        feature_selector,
        k_candidates=None,
        weights_candidates=None,
        cv_splits=3,
        min_train_samples=5,
    ):
        self._feature_selector = feature_selector
        self._k_candidates = k_candidates or [3, 5, 7, 9, 12, 15]
        self._weights_candidates = weights_candidates or ['uniform', 'distance']
        self._cv_splits = cv_splits
        self._min_train_samples = min_train_samples

    @classmethod
    def create_default(cls):
        return cls(feature_selector=LeaveOneOutMseSelector())

    def predict_for_all_users(self, df_train, df_task, movie_ids, matrices):
        df_submission = df_task.copy()
        movie_to_idx = {m_id: i for i, m_id in enumerate(movie_ids)}
        all_components = list(matrices.keys())

        train_grouped = df_train.groupby('user_id')
        unique_users = df_submission['user_id'].unique()

        print(f"Rozpoczynam dobór cech i trenowanie kNN dla {len(unique_users)} użytkowników...")

        for idx, user_id in enumerate(unique_users, 1):
            user_task_mask = df_submission['user_id'] == user_id
            user_task_rows = df_submission[user_task_mask]

            if user_id not in train_grouped.groups:
                df_submission.loc[user_task_mask, 'evaluation'] = 3
                continue

            user_train = train_grouped.get_group(user_id)
            valid_train = user_train[user_train['movie_id'].isin(movie_to_idx)]
            user_mean_rating = int(round(user_train['evaluation'].mean())) if len(user_train) > 0 else 3

            if len(valid_train) < self._min_train_samples:
                df_submission.loc[user_task_mask, 'evaluation'] = user_mean_rating
                continue

            u_train_idx = np.array([movie_to_idx[m] for m in valid_train['movie_id']])
            y_train = valid_train['evaluation'].values

            best_knn, best_dist_matrix = self._fit_best_user_model(
                u_train_idx, y_train, matrices, all_components
            )

            predictions = self._predict_user_movies(
                user_task_rows['movie_id'],
                u_train_idx,
                movie_to_idx,
                best_knn,
                best_dist_matrix,
                user_mean_rating,
            )

            df_submission.loc[user_task_mask, 'evaluation'] = predictions

            if idx % 50 == 0 or idx == len(unique_users):
                print(f"Przetworzono {idx}/{len(unique_users)} użytkowników...")

        df_submission['evaluation'] = df_submission['evaluation'].astype(int)
        return df_submission

    def _fit_best_user_model(self, u_train_idx, y_train, matrices, all_components):
        ranked_features = self._feature_selector.rank_features(u_train_idx, y_train, matrices)

        expert_set = [c for c in ['genres', 'cast_list', 'director', 'vote_average'] if c in matrices]
        candidate_feature_sets = [
            ranked_features[:2],
            ranked_features[:4],
            ranked_features[:6],
            expert_set or ranked_features[:4],
            all_components,
        ]

        max_k = min(max(self._k_candidates), len(u_train_idx) - 2)
        k_options = [k for k in self._k_candidates if k <= max_k] or [1]
        cv = KFold(n_splits=min(self._cv_splits, len(u_train_idx)), shuffle=True, random_state=42)

        best_score = -float('inf')
        best_dist_matrix = None
        best_params = None

        for feat_set in candidate_feature_sets:
            if not feat_set:
                continue

            combined_sim = sum(matrices[c] for c in feat_set) / len(feat_set)
            combined_dist = np.clip(1.0 - combined_sim, 0.0, 1.0)

            X_train_dist = combined_dist[np.ix_(u_train_idx, u_train_idx)]

            knn = KNeighborsRegressor(metric='precomputed')
            param_grid = {
                'n_neighbors': k_options,
                'weights': self._weights_candidates,
            }

            grid = GridSearchCV(knn, param_grid, cv=cv, scoring='neg_mean_squared_error')
            grid.fit(X_train_dist, y_train)

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_dist_matrix = combined_dist
                best_params = grid.best_params_

        final_knn = KNeighborsRegressor(metric='precomputed', **best_params)
        X_train_best = best_dist_matrix[np.ix_(u_train_idx, u_train_idx)]
        final_knn.fit(X_train_best, y_train)

        return final_knn, best_dist_matrix

    @staticmethod
    def _predict_user_movies(task_movie_ids, u_train_idx, movie_to_idx, model, dist_matrix, fallback_rating):
        predictions = []
        for movie_id in task_movie_ids:
            if movie_id in movie_to_idx:
                t_idx = movie_to_idx[movie_id]
                X_test_dist = dist_matrix[[t_idx], :][:, u_train_idx]
                pred = model.predict(X_test_dist)[0]
                predictions.append(int(np.clip(round(pred), 0, 5)))
            else:
                predictions.append(fallback_rating)
        return predictions