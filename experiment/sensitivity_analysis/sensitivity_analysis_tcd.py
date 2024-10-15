import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值
results_tcd = {
    'DT': {
        'AUC': [0.6143, 0.9158, 0.9187, 0.9181, 0.9210],
        'Accuracy': [0.7287, 0.9223, 0.9139, 0.9217, 0.9143],
        'Recall': [0.4085, 0.7223, 0.7995, 0.7685, 0.8222],
        'F1': [0.3999, 0.7765, 0.8096, 0.7952, 0.8161],
        'G-mean': [0.5786, 0.8203, 0.8641, 0.8457, 0.8749]
    },
    'RF': {
        'AUC': [0.7639, 0.9521, 0.9461, 0.9305, 0.9075],
        'Accuracy': [0.8159, 0.9270, 0.8987, 0.8523, 0.7836],
        'Recall': [0.3707, 0.7154, 0.7252, 0.6806, 0.7104],
        'F1': [0.4711, 0.7754, 0.7448, 0.6574, 0.5901],
        'G-mean': [0.5910, 0.8176, 0.8178, 0.7714, 0.7521]
    },
    'XGBoost': {
        'AUC': [0.7661, 0.9004, 0.9039, 0.8931, 0.8739],
        'Accuracy': [0.8143, 0.8710, 0.8709, 0.8582, 0.8115],
        'Recall': [0.3681, 0.5011, 0.6445, 0.6699, 0.7449],
        'F1': [0.4673, 0.6230, 0.6855, 0.6667, 0.6380],
        'G-mean': [0.5885, 0.6919, 0.7721, 0.7732, 0.7840]
    },
    'LightGBM': {
        'AUC': [0.7805, 0.8516, 0.8429, 0.8165, 0.7731],
        'Accuracy': [0.8204, 0.8432, 0.8241, 0.7920, 0.7294],
        'Recall': [0.3703, 0.4135, 0.5119, 0.5100, 0.5814],
        'F1': [0.4770, 0.5360, 0.5627, 0.5169, 0.4876],
        'G-mean': [0.5925, 0.6295, 0.6824, 0.6634, 0.6688]
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
for i, (model, metrics) in enumerate(results_tcd.items()):
    ax = axs[i // 2, i % 2]

    # 绘制五条线
    for metric, values in metrics.items():
        ax.plot(n_values, values, label=metric, marker=markers[metric], markersize=6)

    ax.set_title(model, fontsize=14)
    ax.set_xlabel('Number of Weak Classifiers (n)', fontsize=12)
    ax.set_ylabel('Performance Metric', fontsize=12)
    ax.set_ylim(0.2, 1)  # 设置Y轴范围
    ax.set_xticks(n_values)  # 设置X轴刻度为整数
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=10)

    # 移除右边和上边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# 调整布局
plt.tight_layout()
# 保存图像
plt.savefig('tcd_performance_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
