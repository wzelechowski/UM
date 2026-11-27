import numpy as np
from model.latent_factor_model import LatentFactorModel
from optimizer.joint_gd_optimizer import JointGradientDescentOptimizer
from optimizer.als_optimizer import AlternatingLeastSquaresOptimizer


class CrossValidationValidator:
    def __init__(self, n_splits=4, candidate_configs=None):
        self.n_splits = int(n_splits)
        self.candidate_configs = candidate_configs or [
            {'method': 'als', 'n_factors': 6, 'reg_lambda': 1.0, 'n_epochs': 20, 'mean_norm': True},
            {'method': 'als', 'n_factors': 10, 'reg_lambda': 1.5, 'n_epochs': 20, 'mean_norm': True},
            {'method': 'als', 'n_factors': 16, 'reg_lambda': 2.5, 'n_epochs': 20, 'mean_norm': True},
            {'method': 'als', 'n_factors': 24, 'reg_lambda': 4.0, 'n_epochs': 20, 'mean_norm': True},
            {'method': 'gd', 'n_factors': 12, 'reg_lambda': 0.15, 'lr': 0.015, 'n_epochs': 120, 'mean_norm': True},
            {'method': 'gd', 'n_factors': 20, 'reg_lambda': 0.25, 'lr': 0.015, 'n_epochs': 120, 'mean_norm': True},
            {'method': 'als', 'n_factors': 10, 'reg_lambda': 1.5, 'n_epochs': 20, 'mean_norm': False},
        ]

    def build_optimizer(self, cfg):
        if cfg['method'] == 'als':
            return AlternatingLeastSquaresOptimizer(
                reg_lambda=cfg['reg_lambda'],
                n_epochs=cfg['n_epochs'],
            )
        return JointGradientDescentOptimizer(
            lr=cfg.get('lr', 0.015),
            reg_lambda=cfg['reg_lambda'],
            n_epochs=cfg['n_epochs'],
        )

    def select_best_config(self, n_users, n_movies, u_indices, m_indices, ratings):
        n_samples = len(ratings)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_samples)
        folds = np.array_split(perm, self.n_splits)

        best_mse = float('inf')
        best_cfg = self.candidate_configs[0]

        print(f"Rozpoczynam {self.n_splits}-krotną walidację krzyżową (dobór wymiaru K, regularyzacji lambda i metody optymalizacji)...")

        for cfg in self.candidate_configs:
            fold_mses = []

            for fold_idx in range(self.n_splits):
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != fold_idx])

                model = LatentFactorModel(
                    n_users=n_users,
                    n_movies=n_movies,
                    n_factors=cfg['n_factors'],
                    use_mean_normalization=cfg['mean_norm'],
                    random_seed=42 + fold_idx,
                )
                optimizer = self.build_optimizer(cfg)
                optimizer.fit(model, u_indices[train_idx], m_indices[train_idx], ratings[train_idx])

                preds = model.predict_batch(u_indices[val_idx], m_indices[val_idx])
                preds_clipped = np.clip(preds, 0.0, 5.0)
                mse = float(np.mean((preds_clipped - ratings[val_idx]) ** 2))
                fold_mses.append(mse)

            mean_mse = float(np.mean(fold_mses))
            print(
                f"  Konfiguracja: method={cfg['method']}, K={cfg['n_factors']}, "
                f"lambda={cfg['reg_lambda']}, mean_norm={cfg['mean_norm']} -> CV MSE = {mean_mse:.4f}"
            )

            if mean_mse < best_mse:
                best_mse = mean_mse
                best_cfg = cfg

        print(f"Wybrano najlepszą konfigurację (CV MSE = {best_mse:.4f}): {best_cfg}")
        return best_cfg
