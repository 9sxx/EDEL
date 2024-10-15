import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值
results_ccfd = {
    'DT': {
        'AUC': [0.8921, 0.9767, 0.9797, 0.9827, 0.9827],
        'Accuracy': [0.9992, 0.9998, 0.9998, 0.9998, 0.9997],
        'Recall': [0.7846, 0.8929, 0.9172, 0.9090, 0.9192],
        'F1': [0.7741, 0.9283, 0.9279, 0.9345, 0.9182],
        'G-mean': [0.8856, 0.9419, 0.9561, 0.9516, 0.9570]
    },
    'RF': {
        'AUC': [0.9497, 0.9900, 0.9939, 0.9925, 0.9923],
        'Accuracy': [0.9996, 0.9998, 0.9998, 0.9998, 0.9998],
        'Recall': [0.7846, 0.9111, 0.9212, 0.9090, 0.9131],
        'F1': [0.8588, 0.9453, 0.9475, 0.9420, 0.9370],
        'G-mean': [0.8857, 0.9528, 0.9583, 0.9517, 0.9540]
    },
    'XGBoost': {
        'AUC': [0.9789, 0.9966, 0.9965, 0.9964, 0.9968],
        'Accuracy': [0.9996, 0.9998, 0.9998, 0.9998, 0.9998],
        'Recall': [0.8049, 0.9172, 0.9212, 0.9192, 0.9192],
        'F1': [0.8749, 0.9506, 0.9493, 0.9502, 0.9446],
        'G-mean': [0.8969, 0.9560, 0.9584, 0.9573, 0.9575]
    },
    'LightGBM': {
        'AUC': [0.7549, 0.9774, 0.9948, 0.9944, 0.9965],
        'Accuracy': [0.9959, 0.9994, 0.9997, 0.9998, 0.9998],
        'Recall': [0.5953, 0.6815, 0.8665, 0.9050, 0.9313],
        'F1': [0.3834, 0.7791, 0.8993, 0.9241, 0.9366],
        'G-mean': [0.7681, 0.8215, 0.9277, 0.9488, 0.9639]
    },
}

# 不同指标的 marker 样式
markers = {
    'AUC': 'o',  # 圆形
    'Accuracy': 's',  # 方形
    'Recall': 'D',  # 菱形
    'F1': 'v',  # 倒三角
    'G-mean': '^'  # 上三角
}

# 设置图形样式
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20  # 字体大小设置为20pt
plt.rcParams['axes.linewidth'] = 1.5

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# 模型名称列表
models = ['DT', 'RF', 'XGBoost', 'LightGBM']

# 用于在后面创建全局图例
handles, labels = [], []

# 绘制每个模型的图
for i, model in enumerate(models):
    ax = axs[i // 2, i % 2]

    # 绘制每个指标的图
    for metric, values in results_ccfd[model].items():
        line, = ax.plot(n_values, values, label=metric, marker=markers[metric], markersize=8)
        # 仅在第一次迭代时收集图例句柄和标签
        if i == 0:
            handles.append(line)
            labels.append(metric)

    ax.set_title(f'CCFD - {model}', fontsize=20)

    # 设置不同的Y轴刻度
    if model == 'DT':
        ax.set_ylim(0.7, 1.02)
        ax.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'RF':
        ax.set_ylim(0.7, 1.02)
        ax.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'XGBoost':
        ax.set_ylim(0.7, 1.02)
        ax.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'LightGBM':
        ax.set_ylim(0.3, 1.02)
        ax.set_yticks([0.3, 0.5, 0.7, 0.9, 1])

    ax.set_xticks(n_values)
    ax.set_xlabel('Number of Weak Classifiers (N)', fontsize=20)
    ax.set_ylabel('Performance Metric', fontsize=20)

    # 移除右边和上边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(False)

# 在图的底部显示一次全局图例
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=20)

# 调整子图之间的间距并留出下方图例空间
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.16)

# 保存图像
plt.savefig('sensitivity_analysis_ccfd.pdf', bbox_inches='tight')

plt.show()
