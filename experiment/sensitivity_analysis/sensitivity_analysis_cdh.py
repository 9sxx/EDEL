import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值
results_cdh = {
    'DT': {
        'AUC': [0.5979, 0.9186, 0.9241, 0.9275, 0.9284],
        'Accuracy': [0.7964, 0.9396, 0.9316, 0.9387, 0.9280],
        'Recall': [0.3241, 0.6630, 0.7418, 0.7272, 0.7800],
        'F1': [0.3072, 0.7262, 0.7584, 0.7526, 0.7622],
        'G-mean': [0.5319, 0.7815, 0.8346, 0.8217, 0.8531]
    },
    'RF': {
        'AUC': [0.7974, 0.9549, 0.9506, 0.9447, 0.9365],
        'Accuracy': [0.8600, 0.9425, 0.9024, 0.8569, 0.8014],
        'Recall': [0.1729, 0.6820, 0.7760, 0.8054, 0.8501],
        'F1': [0.2561, 0.7280, 0.6829, 0.6057, 0.5433],
        'G-mean': [0.4098, 0.7899, 0.8377, 0.8294, 0.8195]
    },
    'XGBoost': {
        'AUC': [0.8269, 0.8462, 0.8365, 0.8231, 0.8100],
        'Accuracy': [0.8654, 0.8689, 0.8322, 0.7963, 0.7573],
        'Recall': [0.1706, 0.2849, 0.4945, 0.5735, 0.6561],
        'F1': [0.2610, 0.3749, 0.4508, 0.4390, 0.4296],
        'G-mean': [0.4084, 0.5213, 0.6618, 0.6903, 0.7123]
    },
    'LightGBM': {
        'AUC': [0.8302, 0.8352, 0.8209, 0.8070, 0.7946],
        'Accuracy': [0.8669, 0.8592, 0.8128, 0.7759, 0.7407],
        'Recall': [0.1597, 0.3428, 0.4982, 0.5845, 0.6524],
        'F1': [0.2506, 0.4039, 0.4258, 0.4209, 0.4121],
        'G-mean': [0.3959, 0.5682, 0.6559, 0.6868, 0.7018]
    },
}

# 不同指标的 marker 样式
markers = {
    'AUC': 'o',       # 圆形
    'Accuracy': 's',  # 方形
    'Recall': 'D',    # 菱形
    'F1': 'v',        # 倒三角
    'G-mean': '^'     # 上三角
}

# 设置图形样式
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams['axes.linewidth'] = 1.5

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

# 绘制每个模型的图
for i, (model, metrics) in enumerate(results_cdh.items()):
    ax = axs[i // 2, i % 2]

    # 绘制五条线
    for metric, values in metrics.items():
        ax.plot(n_values, values, label=metric, marker=markers[metric], markersize=6)

    ax.set_title(model, fontsize=14)
    ax.set_xlabel('Number of Weak Classifiers (n)', fontsize=12)
    ax.set_ylabel('Performance Metric', fontsize=12)
    ax.set_ylim(0.1, 1)  # 设置Y轴范围
    ax.set_xticks(n_values)  # 设置X轴刻度为整数
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=10)

    # 移除右边和上边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# 调整布局
plt.tight_layout()
# 保存图像
plt.savefig('cdh_performance_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
