import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值
results_sbd = {
    'DT': {
        'AUC': [0.9030, 0.9896, 0.9911, 0.9918, 0.9923],
        'Accuracy': [0.9063, 0.9757, 0.9770, 0.9750, 0.9735],
        'Recall': [0.8875, 0.9515, 0.9708, 0.9603, 0.9680],
        'F1': [0.8819, 0.9676, 0.9709, 0.9675, 0.9663],
        'G-mean': [0.9028, 0.9708, 0.9759, 0.9722, 0.9725]
    },
    'RF': {
        'AUC': [0.9860, 0.9971, 0.9972, 0.9971, 0.9970],
        'Accuracy': [0.9539, 0.9876, 0.9861, 0.9826, 0.9770],
        'Recall': [0.9272, 0.9763, 0.9835, 0.9818, 0.9846],
        'F1': [0.9406, 0.9840, 0.9824, 0.9779, 0.9711],
        'G-mean': [0.9489, 0.9855, 0.9856, 0.9824, 0.9783]
    },
    'XGBoost': {
        'AUC': [0.9872, 0.9971, 0.9968, 0.9968, 0.9960],
        'Accuracy': [0.9513, 0.9839, 0.9839, 0.9815, 0.9770],
        'Recall': [0.9344, 0.9686, 0.9802, 0.9725, 0.9724],
        'F1': [0.9379, 0.9792, 0.9796, 0.9762, 0.9707],
        'G-mean': [0.9482, 0.9811, 0.9833, 0.9798, 0.9761]
    },
    'LightGBM': {
        'AUC': [0.9882, 0.9973, 0.9969, 0.9969, 0.9965],
        'Accuracy': [0.9565, 0.9848, 0.9841, 0.9828, 0.9800],
        'Recall': [0.9432, 0.9741, 0.9791, 0.9741, 0.9780],
        'F1': [0.9447, 0.9804, 0.9799, 0.9780, 0.9747],
        'G-mean': [0.9541, 0.9828, 0.9832, 0.9812, 0.9796]
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
handles = []
labels = []

# 绘制每个模型的图
for i, model in enumerate(models):
    ax = axs[i // 2, i % 2]

    # 绘制每个指标的图
    for metric, values in results_sbd[model].items():
        line, = ax.plot(n_values, values, label=metric, marker=markers[metric], markersize=8)
        # 收集图例的句柄和标签，只需在第一个模型中收集一次
        if i == 0:
            handles.append(line)
            labels.append(metric)

    ax.set_title(f'SBD - {model}', fontsize=20)

    # 设置不同的Y轴刻度
    if model == 'DT':
        ax.set_ylim(0.85, 1.01)
        ax.set_yticks([0.85, 0.9, 0.95, 1])
    elif model == 'RF':
        ax.set_ylim(0.9, 1.01)
        ax.set_yticks([0.9, 0.95, 1])
    elif model == 'XGBoost':
        ax.set_ylim(0.9, 1.01)
        ax.set_yticks([0.9, 0.95, 1])
    elif model == 'LightGBM':
        ax.set_ylim(0.9, 1.01)
        ax.set_yticks([0.9, 0.95, 1])

    ax.set_xticks(n_values)
    ax.set_xlabel('Number of Weak Classifiers (N)', fontsize=20)
    ax.set_ylabel('Performance Metric', fontsize=20)

    # 移除右边和上边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(False)

# 在图的底部显示一次全局图例
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=20)

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.16)

# 保存图像
plt.savefig('sensitivity_analysis_sbd.pdf', bbox_inches='tight')

plt.show()
