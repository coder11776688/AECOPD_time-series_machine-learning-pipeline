import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import catboost as cb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from feature_selector import FeatureSelector

warnings.filterwarnings('ignore')

def generate_colors(n):
    import colorsys

    def hsv_to_hex(h, s, v):
        rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.1
        colors.append(hsv_to_hex(hue, saturation, value))
    return colors

class ModelPipeline:
    def __init__(self, config_path, tune_params=False, search_method='grid', feature_selection=None,
                 output_dir=None):
        self.tune_params = tune_params
        self.search_method = search_method
        self.feature_selection = feature_selection
        self.load_config(config_path)
        if self.config['feature_selection'].get('enabled', False):
            self.feature_selection = self.config['feature_selection']['method']

        self.model_classes = {
            'KNN': KNeighborsClassifier,
            'CART': DecisionTreeClassifier,
            'Random Forest': RandomForestClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'CatBoost': cb.CatBoostClassifier,
            'XGBoost': xgb.XGBClassifier,
            'Logistic Regression': LogisticRegression,
            'LDA': LinearDiscriminantAnalysis,
            'QDA': QuadraticDiscriminantAnalysis,
            'SVM': SVC,
            'Naive Bayes': GaussianNB,
            'Lasso': lambda: LogisticRegression(penalty='l1', solver='liblinear'),
            'Ridge': lambda: LogisticRegression(penalty='l2', solver='lbfgs'),
            'Elastic Net': lambda: LogisticRegression(penalty='elasticnet', solver='saga')
        }

        self.results = {
            'Model': [],
            'Dataset': [],
            'Accuracy': [],
            'F1 Score': [],
            'Precision': [],
            'Recall': [],
            'Specificity': [],
            'AUC': [],
            'Best Parameters': []
        }

        self.confusion_matrices = {
            'Model': [],
            'Dataset': [],
            'TN': [],
            'FP': [],
            'FN': [],
            'TP': []
        }

        self.cv_metrics = {
            'Model': [],
            'Fold': [],
            'Dataset': [],
            'Accuracy': [],
            'F1 Score': [],
            'Precision': [],
            'Recall': [],
            'Specificity': [],
            'AUC': []
        }

        self.cv_metrics_summary = {
            'Model': [],
            'Dataset': [],
            'Metric': [],
            'Mean': [],
            'Std': []
        }

        self.roc_data = {
            'model_name': [],
            'fpr': [],
            'tpr': [],
            'auc': []
        }

        if output_dir is None:
            feature_sel = self.feature_selection if self.feature_selection else "no_feature_selection"
            output_dir = f"results_seed{self.config['global_settings']['random_state']}_{feature_sel}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.initialize_models()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.global_settings = self.config['global_settings']

    def initialize_models(self):
        self.models = {}
        self.param_grids = {}
        for model_name, model_config in self.config['models'].items():
            if model_config['enabled']:
                if model_name == 'CatBoost':
                    model = self.model_classes[model_name](verbose=False,
                                                           random_state=self.global_settings['random_state'])
                elif model_name == 'SVM':
                    model = self.model_classes[model_name](probability=True,
                                                           random_state=self.global_settings['random_state'])
                elif hasattr(self.model_classes[model_name], 'random_state'):
                    model = self.model_classes[model_name](random_state=self.global_settings['random_state'])
                else:
                    model = self.model_classes[model_name]()
                self.models[model_name] = model
                self.param_grids[model_name] = model_config['params']

    def calculate_specificity(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0

    def evaluate_model(self, model, X, y, dataset_type, full_model_name):
        try:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            spec = self.calculate_specificity(y, y_pred)
            rocauc = roc_auc_score(y, y_prob)

            self.results['Accuracy'].append(acc)
            self.results['F1 Score'].append(f1)
            self.results['Precision'].append(prec)
            self.results['Recall'].append(rec)
            self.results['Specificity'].append(spec)
            self.results['AUC'].append(rocauc)

            self.save_confusion_matrix_counts(y, y_pred, dataset_type, full_model_name)
        except Exception as e:
            import traceback
            print(f"Error evaluating {model.__class__.__name__}: {str(e)}")
            traceback.print_exc()
            for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity', 'AUC']:
                self.results[metric].append(None)

    def save_confusion_matrix_counts(self, y_true, y_pred, dataset_type, full_model_name):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = None
        self.confusion_matrices['Model'].append(full_model_name)
        self.confusion_matrices['Dataset'].append(dataset_type)
        self.confusion_matrices['TN'].append(tn)
        self.confusion_matrices['FP'].append(fp)
        self.confusion_matrices['FN'].append(fn)
        self.confusion_matrices['TP'].append(tp)

    def tune_model(self, model_name, model, param_grid, X_train, y_train):
        print(f"Tuning hyperparameters for {model_name}...")

        if not param_grid:
            print(f"No parameters to tune for {model_name}, using default parameters.")
            model.fit(X_train, y_train)
            return model, None
        _scoring = 'f1'
        if self.search_method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.global_settings['cv_folds'],
                scoring=_scoring,
                n_jobs=self.global_settings['n_jobs']
            )
        else:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,
                cv=self.global_settings['cv_folds'],
                scoring=_scoring,
                n_jobs=self.global_settings['n_jobs'],
                random_state=self.global_settings['random_state']
            )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_

    def cross_val_metrics(self, model, X, y, model_name):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.global_settings['random_state'])
        fold_idx = 1

        train_metrics = {k: [] for k in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity', 'AUC']}
        val_metrics = {k: [] for k in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity', 'AUC']}

        for train_index, val_index in cv.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            try:
                model_clone = type(model)(**model.get_params())
            except Exception:
                model_clone = model
            model_clone.fit(X_train, y_train)

            y_train_pred = model_clone.predict(X_train)
            if hasattr(model_clone, 'predict_proba'):
                y_train_prob = model_clone.predict_proba(X_train)[:, 1]
            else:
                y_train_prob = np.zeros_like(y_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_prec = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_spec = self.calculate_specificity(y_train, y_train_pred)
            train_auc = roc_auc_score(y_train, y_train_prob) if len(set(y_train)) == 2 else np.nan

            self.cv_metrics['Model'].append(model_name)
            self.cv_metrics['Fold'].append(fold_idx)
            self.cv_metrics['Dataset'].append('Train')
            self.cv_metrics['Accuracy'].append(train_acc)
            self.cv_metrics['F1 Score'].append(train_f1)
            self.cv_metrics['Precision'].append(train_prec)
            self.cv_metrics['Recall'].append(train_recall)
            self.cv_metrics['Specificity'].append(train_spec)
            self.cv_metrics['AUC'].append(train_auc)

            for k, v in zip(['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity', 'AUC'],
                            [train_acc, train_f1, train_prec, train_recall, train_spec, train_auc]):
                train_metrics[k].append(v)

            y_val_pred = model_clone.predict(X_val)
            if hasattr(model_clone, 'predict_proba'):
                y_val_prob = model_clone.predict_proba(X_val)[:, 1]
            else:
                y_val_prob = np.zeros_like(y_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_prec = precision_score(y_val, y_val_pred)
            val_recall = recall_score(y_val, y_val_pred)
            val_spec = self.calculate_specificity(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_prob) if len(set(y_val)) == 2 else np.nan

            self.cv_metrics['Model'].append(model_name)
            self.cv_metrics['Fold'].append(fold_idx)
            self.cv_metrics['Dataset'].append('Validation')
            self.cv_metrics['Accuracy'].append(val_acc)
            self.cv_metrics['F1 Score'].append(val_f1)
            self.cv_metrics['Precision'].append(val_prec)
            self.cv_metrics['Recall'].append(val_recall)
            self.cv_metrics['Specificity'].append(val_spec)
            self.cv_metrics['AUC'].append(val_auc)

            for k, v in zip(['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity', 'AUC'],
                            [val_acc, val_f1, val_prec, val_recall, val_spec, val_auc]):
                val_metrics[k].append(v)

            fold_idx += 1

        for ds_metrics, dset in zip([train_metrics, val_metrics], ['Train', 'Validation']):
            for metric in ds_metrics.keys():
                vals = ds_metrics[metric]
                self.cv_metrics_summary['Model'].append(model_name)
                self.cv_metrics_summary['Dataset'].append(dset)
                self.cv_metrics_summary['Metric'].append(metric)
                self.cv_metrics_summary['Mean'].append(np.mean(vals))
                self.cv_metrics_summary['Std'].append(np.std(vals))

    def train_and_evaluate(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.global_settings['test_size'],
            random_state=self.global_settings['random_state']
        )
        random_state = self.global_settings['random_state']

        if self.feature_selection:
            feature_selector = FeatureSelector(
                method=self.feature_selection,
                config=self.config['feature_selection'],
                random_state=self.global_settings['random_state']
            )
            X_train = feature_selector.fit_transform(X_train, y_train)
            X_test = feature_selector.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        plt.figure(figsize=(10, 8))
        colors = generate_colors(len(self.models))

        for (model_name, model), color in zip(self.models.items(), colors):
            print(f"\nProcessing {model_name}...")
            full_model_name = f"{self.feature_selection.upper()}_{model_name}" if self.feature_selection else model_name
            try:
                best_params = None
                if self.tune_params:
                    model, best_params = self.tune_model(
                        model_name,
                        model,
                        self.param_grids[model_name],
                        X_train_scaled,
                        y_train
                    )
                else:
                    model.fit(X_train_scaled, y_train)

                self.cross_val_metrics(model, X_train_scaled, y_train, full_model_name)

                self.results['Model'].append(full_model_name)
                self.results['Dataset'].append('Train')
                self.results['Best Parameters'].append(str(best_params) if best_params else 'Default')
                self.evaluate_model(model, X_train_scaled, y_train, 'Train', full_model_name)

                self.results['Model'].append(full_model_name)
                self.results['Dataset'].append('Test')
                self.results['Best Parameters'].append(str(best_params) if best_params else 'Default')
                self.evaluate_model(model, X_test_scaled, y_test, 'Test', full_model_name)

                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    self.roc_data['model_name'].append(model_name)
                    self.roc_data['fpr'].append(fpr)
                    self.roc_data['tpr'].append(tpr)
                    self.roc_data['auc'].append(roc_auc)
                    plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                import traceback
                print(f"Error with {model_name}: {str(e)}")
                traceback.print_exc()
                continue

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        roc_img_path = os.path.join(
            self.output_dir,
            f'roc_curves_seed_{random_state}_feature_{self.feature_selection}.png'
        )
        plt.savefig(roc_img_path, bbox_inches='tight', dpi=300)
        plt.close()

        results_df = pd.DataFrame(self.results)
        feature_selection_str = f"_{self.feature_selection}" if self.feature_selection else ""
        csv_path = os.path.join(
            self.output_dir,
            f'model_evaluation_results{feature_selection_str}_seed{self.global_settings["random_state"]}.csv'
        )
        results_df.to_csv(csv_path, index=False)

        cm_counts_df = pd.DataFrame(self.confusion_matrices)
        cm_csv_path = os.path.join(self.output_dir, "confusion_matrix_counts.csv")
        cm_counts_df.to_csv(cm_csv_path, index=False)

        cv_metrics_df = pd.DataFrame(self.cv_metrics)
        cv_metrics_stats_df = pd.DataFrame(self.cv_metrics_summary)
        cv_csv_path = os.path.join(self.output_dir, '10_fold_crossvalidation_metrics.csv')
        with pd.ExcelWriter(cv_csv_path.replace(".csv", ".xlsx")) as writer:
            cv_metrics_df.to_excel(writer, sheet_name="FoldMetrics", index=False)
            cv_metrics_stats_df.to_excel(writer, sheet_name="MetricSummary", index=False)
        cv_metrics_df.to_csv(cv_csv_path, index=False)

        print(f"\nResults (including confusion matrix csv and 10-fold CV) have been saved in '{self.output_dir}/'")
        print(f"Main evaluation CSV: {csv_path}")
        print(f"Confusion matrix counts: {cm_csv_path}")
        print(f"10-fold crossvalidation metrics: {cv_csv_path}")
        return results_df

if __name__ == "__main__":
    df = pd.read_csv("./data/data_Incremental.csv")

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0].values

    from sklearn.preprocessing import LabelEncoder

    def label_encode_feature(df, column_name):
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])
        return df

    for col in ["jieqi", "Sex"]:
        if col in X.columns:
            X = label_encode_feature(X, col)

    X = X.values

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("unique labels:", np.unique(y))

    pipeline_lasso = ModelPipeline(
         config_path='model_config_feature_sel_lasso.yaml',
         tune_params=True,
         search_method='grid',
     )

    pipeline_woa = ModelPipeline(
        config_path='model_config_feature_sel_woa.yaml',
        tune_params=True,
        search_method='grid',
    )

    pipeline_noneselection = ModelPipeline(
         config_path='model_config.yaml',
         tune_params=True,
         search_method='grid',
     )

    results_lasso = pipeline_lasso.train_and_evaluate(X, y)

    results_woa = pipeline_woa.train_and_evaluate(X, y)

    results_noneselection = pipeline_noneselection.train_and_evaluate(X, y)