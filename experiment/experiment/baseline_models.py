import pandas as pd
import lightgbm as lgb
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix


def evaluate_model(classifier, X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化存储分数的列表
    metrics_summary = {
        'test_auc': [], 'test_accuracy': [],
        'test_recall': [], 'test_f1': [],
        'test_gmean': []
    }

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型
        classifier.fit(X_train, y_train)

        # 对验证集的预测标签
        y_test_pred = classifier.predict(X_test)
        y_test_pred_proba = classifier.predict_proba(X_test)

        # 计算各项指标
        metrics_summary['test_auc'].append(roc_auc_score(y_test, y_test_pred_proba[:, 1]))
        metrics_summary['test_accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics_summary['test_recall'].append(recall_score(y_test, y_test_pred))
        metrics_summary['test_f1'].append(f1_score(y_test, y_test_pred))

        # 计算G-mean
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp)
        gmean = np.sqrt(specificity * recall_score(y_test, y_test_pred))
        metrics_summary['test_gmean'].append(gmean)

    # 打印平均指标
    print('#' * 20)
    metrics_results = {}
    for metric in metrics_summary:
        mean_value = np.mean(metrics_summary[metric])
        std_value = np.std(metrics_summary[metric], ddof=1)  # 用于样本标准差的计算

        print(f'Mean {metric}: {mean_value:.4f}, Std {metric}: {std_value:.4f}')

        # 保留小数点后四位
        mean_value = np.around(mean_value, 4)
        std_value = np.around(std_value, 4)

        # 存储均值和标准差，保留四位小数
        metrics_results[metric] = {
            'mean': mean_value,
            'std': std_value
        }

    # 返回包括均值和标准差的指标
    return metrics_results


# 加载处理后的特征数据和标签
data = pd.read_csv('CDC Diabetes Health Indicators Dataset/data.csv')
# 分离特征和标签
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

# 初始化分类器
classifiers = {
    "LightGBM": lgb.LGBMClassifier(verbosity=-1, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# 存储结果
results = []

# 对每个分类器进行评估
for name, clf in classifiers.items():
    print(f"=================={name}===================")
    metrics = evaluate_model(clf, X, y)
    metrics['Classifier'] = name
    results.append(metrics)

# 转换为DataFrame以便查看
results_df = pd.DataFrame(results)

# 显示最终结果
print(results_df)
