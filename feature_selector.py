# feature_selector.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WhaleOptimizer:
    def __init__(self, n_whales, max_iter, n_features, fitness_func, b=1, a_decrease_factor=0.5):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.b = b
        self.a_decrease_factor = a_decrease_factor

    def initialize_population(self):
        return np.random.randint(2, size=(self.n_whales, self.n_features))

    def calculate_fitness(self, positions):
        return np.array([self.fitness_func(pos) for pos in positions])

    def optimize(self):
        positions = self.initialize_population()
        fitness = self.calculate_fitness(positions)

        best_whale_idx = np.argmax(fitness)
        best_position = positions[best_whale_idx].copy()
        best_fitness = fitness[best_whale_idx]

        for iter in range(self.max_iter):
            a = 2 * (1 - iter / self.max_iter)

            for i in range(self.n_whales):
                r1, r2, r3, p = np.random.random(4)

                if p < 0.5:
                    if r1 < 0.5:
                        l = np.random.random()
                        D = np.abs(best_position - positions[i])
                        positions[i] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_position
                    else:
                        A = 2 * a * r2 - a
                        C = 2 * r3
                        D = np.abs(C * best_position - positions[i])
                        positions[i] = best_position - A * D
                else:
                    rand_whale_idx = np.random.randint(0, self.n_whales)
                    rand_position = positions[rand_whale_idx]
                    A = 2 * a * r2 - a
                    C = 2 * r3
                    D = np.abs(C * rand_position - positions[i])
                    positions[i] = rand_position - A * D

                positions[i] = np.where(self.sigmoid(positions[i]) > 0.5, 1, 0)

            fitness = self.calculate_fitness(positions)
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = positions[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            if iter % 10 == 0:  # 每10次迭代记录一次进度
                logging.info(f"WOA iteration {iter}/{self.max_iter}, best fitness: {best_fitness:.4f}")

        return best_position, best_fitness

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class FeatureSelector:
    def __init__(self, method, config, random_state=42):
        """
        method: 'rf', 'woa', 或 'lasso'
        config: 配置字典，包含特征选择的参数
        """
        self.method = method
        self.config = config
        self.random_state = random_state
        self.selected_features_mask = None
        self.feature_importance = None
        self.selected_features_names = None
        self.selected_features_indices = None

        # 验证输入参数
        if method not in ['rf', 'woa', 'lasso']:
            raise ValueError("Method must be 'rf' 'woa' or 'lasso'")
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

    def select_features_rf(self, X, y):
        """使用Random Forest进行特征选择"""
        logging.info("Performing Random Forest feature selection...")

        rf_settings = self.config.get('rf_settings', {})

        # 初始化Random Forest，使用类的random_state
        rf = RandomForestClassifier(
            n_estimators=rf_settings.get('n_estimators', 100),
            max_depth=rf_settings.get('max_depth', None),
            min_samples_split=rf_settings.get('min_samples_split', 2),
            random_state=self.random_state,  # 使用类的random_state
            n_jobs=-1
        )

        # 训练模型
        rf.fit(X, y)

        # 获取特征重要性
        self.feature_importance = rf.feature_importances_

        # 根据阈值选择特征
        importance_threshold = rf_settings.get('importance_threshold', 0.01)
        selected_features_mask = self.feature_importance > importance_threshold
        selected_features = np.where(selected_features_mask)[0]

        # 验证选择的特征
        if len(selected_features) == 0:
            # 如果没有特征超过阈值，至少选择最重要的几个特征
            n_min_features = rf_settings.get('min_features', 5)
            selected_features = np.argsort(self.feature_importance)[-n_min_features:]
            selected_features_mask = np.zeros_like(self.feature_importance, dtype=bool)
            selected_features_mask[selected_features] = True

        logging.info(f"Random Forest selected {len(selected_features)} features")

        # 评估选择的特征
        X_selected = X[:, selected_features]
        cv_scores = cross_val_score(
            RandomForestClassifier(random_state=self.random_state),  # 这里也使用类的random_state
            X_selected,
            y,
            cv=5,
            n_jobs=-1
        )
        logging.info(
            f"Cross-validation score with selected features: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return selected_features, selected_features_mask
    def feature_selection_fitness(self, feature_mask, X, y):
        """WOA特征选择的适应度函数"""
        if np.sum(feature_mask) == 0:
            return 0

        X_np = X if isinstance(X, np.ndarray) else X.values
        y_np = y if isinstance(y, np.ndarray) else y.values

        X_selected = X_np[:, feature_mask == 1]

        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X_selected):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            model = RandomForestClassifier(random_state=self.random_state)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        mean_score = np.mean(cv_scores)
        feature_penalty = 0.001 * np.sum(feature_mask) / len(feature_mask)

        return mean_score - feature_penalty

    def select_features_woa(self, X, y):
        """使用WOA进行特征选择"""
        logging.info("Starting WOA feature selection...")

        if 'woa_settings' not in self.config:
            raise ValueError("WOA settings not found in config")

        woa_settings = self.config['woa_settings']

        woa = WhaleOptimizer(
            n_whales=woa_settings.get('n_whales', 10),
            max_iter=woa_settings.get('max_iter', 50),
            n_features=X.shape[1],
            fitness_func=lambda x: self.feature_selection_fitness(x, X, y),
            b=woa_settings.get('b', 1),
            a_decrease_factor=woa_settings.get('a_decrease_factor', 0.5)
        )

        best_feature_mask, best_fitness = woa.optimize()
        selected_features = np.where(best_feature_mask == 1)[0]

        logging.info(f"WOA selected {len(selected_features)} features with fitness: {best_fitness:.4f}")

        return selected_features, best_feature_mask

    def select_features_lasso(self, X, y):
        """使用Lasso进行特征选择"""
        logging.info("Starting Lasso feature selection...")

        if 'lasso_settings' not in self.config:
            raise ValueError("Lasso settings not found in config")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(
            cv=5,
            random_state=self.random_state,
            max_iter=10000,
            n_jobs=-1
        )

        lasso.fit(X_scaled, y)

        feature_importance = np.abs(lasso.coef_)
        self.feature_importance = feature_importance

        threshold = self.config['lasso_settings']['importance_threshold']
        selected_features_mask = feature_importance > threshold
        selected_features = np.where(selected_features_mask)[0]

        logging.info(f"Lasso selected {len(selected_features)} features")

        return selected_features, selected_features_mask

    def fit(self, X, y):
        """特征选择训练"""
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        if self.method == 'rf':
            self.selected_features_indices, self.selected_features_mask = self.select_features_rf(X_np, y_np)
        elif self.method == 'woa':
            self.selected_features_indices, self.selected_features_mask = self.select_features_woa(X_np, y_np)
        elif self.method == 'lasso':
            self.selected_features_indices, self.selected_features_mask = self.select_features_lasso(X_np, y_np)

        if isinstance(X, pd.DataFrame):
            self.selected_features_names = X.columns[self.selected_features_indices].tolist()

        self.save_feature_selection_results(X, self.selected_features_mask)

        return self

    def transform(self, X):
        """应用特征选择"""
        if self.selected_features_indices is None:
            raise ValueError("Must call fit() before transform()")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_indices]
        return X[:, self.selected_features_indices]

    def fit_transform(self, X, y):
        """组合fit和transform操作"""
        return self.fit(X, y).transform(X)

    def save_feature_selection_results(self, X, feature_mask):
        """保存特征选择结果"""
        try:
            results = {
                'method': self.method,
                'n_original_features': int(len(feature_mask)),
                'n_selected_features': int(np.sum(feature_mask)),
                'selected_features_mask': feature_mask.tolist(),
            }

            if self.selected_features_names is not None:
                results['selected_features_names'] = self.selected_features_names

            if self.method == 'lasso' and self.feature_importance is not None:
                results['feature_importance'] = self.feature_importance.tolist()

            if hasattr(self, 'selected_features_indices'):
                results['selected_features_indices'] = self.selected_features_indices.tolist()

            if isinstance(X, pd.DataFrame):
                results['original_feature_names'] = X.columns.tolist()

            def convert_to_native_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_native_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                return obj

            results = convert_to_native_types(results)

            with open(f'feature_selection_results_{self.method}.json', 'w') as f:
                json.dump(results, f, indent=4)

            logging.info(f"Feature selection results saved to feature_selection_results_{self.method}.json")

        except Exception as e:
            logging.error(f"Error saving feature selection results: {str(e)}")
            raise