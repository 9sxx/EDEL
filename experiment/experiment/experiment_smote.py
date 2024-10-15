import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix


def evaluate_model(classifier, X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_summary = {
        'test_auc': [], 'test_accuracy': [],
        'test_recall': [], 'test_f1': [],
        'test_gmean': []
    }

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        classifier.fit(X_train_resampled, y_train_resampled)

        y_test_pred = classifier.predict(X_test)
        y_test_pred_proba = classifier.predict_proba(X_test)

        metrics_summary['test_auc'].append(roc_auc_score(y_test, y_test_pred_proba[:, 1]))
        metrics_summary['test_accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics_summary['test_recall'].append(recall_score(y_test, y_test_pred))
        metrics_summary['test_f1'].append(f1_score(y_test, y_test_pred))

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp)
        gmean = np.sqrt(specificity * recall_score(y_test, y_test_pred))
        metrics_summary['test_gmean'].append(gmean)

    print('#' * 20)
    metrics_results = {}
    for metric in metrics_summary:
        mean_value = np.mean(metrics_summary[metric])
        std_value = np.std(metrics_summary[metric], ddof=1)

        print(f'Mean {metric}: {mean_value:.4f}, Std {metric}: {std_value:.4f}')

        mean_value = np.around(mean_value, 4)
        std_value = np.around(std_value, 4)

        metrics_results[metric] = {
            'mean': mean_value,
            'std': std_value
        }

    return metrics_results


data = pd.read_csv('Credit Card Fraud Detection/data.csv')

X = data.drop('Class', axis=1)
y = data['Class']

classifiers = {
    "LightGBM": lgb.LGBMClassifier(verbosity=-1, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = []

for name, clf in classifiers.items():
    print(f"=================={name}===================")
    metrics = evaluate_model(clf, X, y)
    metrics['Classifier'] = name
    results.append(metrics)

results_df = pd.DataFrame(results)

print(results_df)
