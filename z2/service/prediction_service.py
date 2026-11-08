import numpy as np
from tree.binary_decision_tree import BinaryDecisionTree
from tree.random_forest import RandomForest
from tree.tree_visualizer import TreeVisualizer
from tree.criterion import GiniCriterion
from service.cross_validation_validator import CrossValidationValidator


class PredictionService:
    def __init__(
        self,
        validator,
        visualizer,
        criterion=None,
        tree_candidates=None,
        forest_candidates=None,
        min_train_samples=5,
    ):
        self._validator = validator
        self._visualizer = visualizer
        self._criterion = criterion or GiniCriterion()
        self._tree_candidates = tree_candidates or [
            {'max_depth': 3, 'min_samples_split': 8, 'min_samples_leaf': 3, 'min_impurity_decrease': 0.005},
            {'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 2, 'min_impurity_decrease': 0.002},
            {'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'min_impurity_decrease': 0.001},
            {'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0},
        ]
        self._forest_candidates = forest_candidates or [
            {'n_estimators': 35, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features_ratio': 'sqrt', 'voting': 'soft'},
            {'n_estimators': 45, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features_ratio': 'sqrt', 'voting': 'soft'},
            {'n_estimators': 45, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features_ratio': 'sqrt', 'voting': 'hard'},
        ]
        self._min_train_samples = min_train_samples

    @classmethod
    def create_default(cls):
        criterion = GiniCriterion()
        return cls(
            validator=CrossValidationValidator(cv_splits=3, criterion=criterion),
            visualizer=TreeVisualizer(),
            criterion=criterion,
        )

    def predict_for_all_users(
        self,
        df_train,
        df_task,
        df_features,
        condition_factories,
        tree_image_path=None,
    ):
        df_sub_tree = df_task.copy()
        df_sub_forest = df_task.copy()

        movie_dict = df_features.to_dict('index')
        train_grouped = df_train.groupby('user_id')
        unique_users = df_task['user_id'].unique()

        tree_visualized = False

        print(f"Rozpoczynam dobór parametrów (CV) oraz trenowanie Drzew i Lasów dla {len(unique_users)} użytkowników...")

        for idx, user_id in enumerate(unique_users, 1):
            user_task_mask = df_task['user_id'] == user_id
            user_task_rows = df_task[user_task_mask]

            if user_id not in train_grouped.groups:
                df_sub_tree.loc[user_task_mask, 'evaluation'] = 3
                df_sub_forest.loc[user_task_mask, 'evaluation'] = 3
                continue

            user_train = train_grouped.get_group(user_id)
            valid_train = user_train[user_train['movie_id'].isin(df_features.index)]
            fallback_rating = int(round(user_train['evaluation'].mean())) if len(user_train) > 0 else 3

            if len(valid_train) < self._min_train_samples:
                df_sub_tree.loc[user_task_mask, 'evaluation'] = fallback_rating
                df_sub_forest.loc[user_task_mask, 'evaluation'] = fallback_rating
                continue

            df_user_X = df_features.loc[valid_train['movie_id']].reset_index(drop=True)
            y_train = valid_train['evaluation'].to_numpy(dtype=int)

            best_tree_params = self._validator.select_best_tree_params(
                df_user_X, y_train, condition_factories, self._tree_candidates
            )
            single_tree = BinaryDecisionTree(
                criterion=self._criterion,
                random_state=42 + int(user_id),
                **best_tree_params,
            )
            single_tree.fit(df_user_X, y_train, condition_factories)

            if not tree_visualized and tree_image_path and single_tree.root.get_depth() > 1:
                self._visualizer.save_tree_png(single_tree, tree_image_path, user_id=user_id)
                print(f"Wygenerowano wizualizację drzewa dla użytkownika {user_id} -> {tree_image_path}")
                tree_visualized = True

            best_forest_params = self._validator.select_best_forest_params(
                df_user_X, y_train, condition_factories, self._forest_candidates
            )
            random_forest = RandomForest(
                criterion=self._criterion,
                random_state=1000 + int(user_id),
                **best_forest_params,
            )
            random_forest.fit(df_user_X, y_train, condition_factories)

            preds_tree = self._predict_user_movies(
                user_task_rows['movie_id'], movie_dict, single_tree, fallback_rating
            )
            preds_forest = self._predict_user_movies(
                user_task_rows['movie_id'], movie_dict, random_forest, fallback_rating
            )

            df_sub_tree.loc[user_task_mask, 'evaluation'] = preds_tree
            df_sub_forest.loc[user_task_mask, 'evaluation'] = preds_forest

            if idx % 20 == 0 or idx == len(unique_users):
                print(f"Przetworzono {idx}/{len(unique_users)} użytkowników...")

        if not tree_visualized and tree_image_path and 'single_tree' in locals():
            self._visualizer.save_tree_png(single_tree, tree_image_path, user_id=unique_users[0])

        df_sub_tree['evaluation'] = df_sub_tree['evaluation'].astype(int)
        df_sub_forest['evaluation'] = df_sub_forest['evaluation'].astype(int)

        return df_sub_tree, df_sub_forest

    @staticmethod
    def _predict_user_movies(task_movie_ids, movie_dict, model, fallback_rating):
        predictions = []
        for movie_id in task_movie_ids:
            if movie_id in movie_dict:
                pred = model.predict_row(movie_dict[movie_id])
                predictions.append(int(np.clip(round(pred), 0, 5)))
            else:
                predictions.append(fallback_rating)
        return predictions
