import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值
results_gmsc = {
    'DT': {
        'AUC': [0.6120, 0.9283, 0.9369, 0.9425, 0.9457],
        'Accuracy': [0.8976, 0.9725, 0.9677, 0.9726, 0.9672],
        'Recall': [0.2802, 0.6750, 0.7415, 0.7354, 0.7817],
        'F1': [0.2679, 0.7267, 0.7562, 0.7594, 0.7690],
        'G-mean': [0.5137, 0.7834, 0.8351, 0.8265, 0.8607]
    },
    'RF': {
        'AUC': [0.8387, 0.9669, 0.9656, 0.9625, 0.9583],
        'Accuracy': [0.9354, 0.9748, 0.9657, 0.9526, 0.9321],
        'Recall': [0.1875, 0.6758, 0.7251, 0.7249, 0.7625],
        'F1': [0.2796, 0.7229, 0.7135, 0.6520, 0.5929],
        'G-mean': [0.4306, 0.7790, 0.8224, 0.8201, 0.8393]
    },
    'XGBoost': {
        'AUC': [0.8573, 0.8992, 0.8999, 0.8973, 0.8963],
        'Accuracy': [0.9356, 0.9456, 0.9440, 0.9393, 0.9282],
        'Recall': [0.1953, 0.2717, 0.3772, 0.4047, 0.4993],
        'F1': [0.2884, 0.3972, 0.4732, 0.4670, 0.4821],
        'G-mean': [0.4393, 0.5146, 0.6059, 0.6234, 0.6898]
    },
    'LightGBM': {
        'AUC': [0.8649, 0.8849, 0.8821, 0.8768, 0.8726],
        'Accuracy': [0.9373, 0.9402, 0.9319, 0.9207, 0.9070],
        'Recall': [0.1911, 0.2459, 0.3372, 0.3782, 0.4621],
        'F1': [0.2895, 0.3534, 0.3972, 0.3884, 0.3988],
        'G-mean': [0.4350, 0.4915, 0.5719, 0.6013, 0.6583]
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
for i, (model, metrics) in enumerate(results_gmsc.items()):
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
plt.savefig('gmsc_performance_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
