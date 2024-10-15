import matplotlib.pyplot as plt

# 数据设置
n_values = [1, 2, 3, 4, 5]  # n的值

# Spambase 数据集的结果
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

# Credit Card Fraud 数据集的结果
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
    'AUC': 'o',       # 圆形
    'Accuracy': 's',  # 方形
    'Recall': 'D',    # 菱形
    'F1': 'v',        # 倒三角
    'G-mean': '^'     # 上三角
}

import matplotlib.pyplot as plt

# 设置图形样式
plt.rcParams["font.family"] = "Times New Roman"  # 使用 Times New Roman 字体
plt.rcParams["font.size"] = 18  # 字体大小设置为18pt
plt.rcParams['axes.linewidth'] = 1.5  # 线条宽度增加

# 创建图形，增加尺寸
fig, axs = plt.subplots(4, 2, figsize=(14, 18))

# 模型名称的列表
models = ['DT', 'RF', 'XGBoost', 'LightGBM']

# 绘制每个模型的图
for i, model in enumerate(models):
    # Spambase 数据集（上方四个子图）
    ax_sbd = axs[i // 2, i % 2]  # 使用整除和取模运算排列前四个子图
    for metric, values in results_sbd[model].items():
        ax_sbd.plot(n_values, values, label=metric, marker=markers[metric], linestyle='-', markersize=8)
    ax_sbd.set_title(f'SBD - {model}', fontsize=18)

    # 设置不同的Y轴刻度
    if model == 'DT':
        ax_sbd.set_ylim(0.85, 1.01)
        ax_sbd.set_yticks([0.85, 0.9, 0.95, 1])
    elif model == 'RF':
        ax_sbd.set_ylim(0.9, 1.01)
        ax_sbd.set_yticks([0.9, 0.95, 1])
    elif model == 'XGBoost':
        ax_sbd.set_ylim(0.9, 1.01)
        ax_sbd.set_yticks([0.9, 0.95, 1])
    elif model == 'LightGBM':
        ax_sbd.set_ylim(0.9, 1.01)
        ax_sbd.set_yticks([0.9, 0.95, 1])

    ax_sbd.set_xticks(n_values)
    if i // 2 == 1:  # 仅在底部图设置X轴标签
        ax_sbd.set_xlabel('Number of Weak Classifiers (N)', fontsize=18)
    ax_sbd.set_ylabel('Metric', fontsize=18)

    # 更改边框线为灰色
    ax_sbd.spines['left'].set_color('gray')
    ax_sbd.spines['left'].set_alpha(0.5)
    ax_sbd.spines['bottom'].set_color('gray')
    ax_sbd.spines['bottom'].set_alpha(0.5)

    ax_sbd.spines['right'].set_visible(False)
    ax_sbd.spines['top'].set_visible(False)

    # Credit Card Fraud 数据集（下方四个子图）
    ax_ccfd = axs[(i // 2) + 2, i % 2]  # 下方的子图从第2行开始
    for metric, values in results_ccfd[model].items():
        ax_ccfd.plot(n_values, values, label=metric, marker=markers[metric], linestyle='--', markersize=8)
    ax_ccfd.set_title(f'CCFD - {model}', fontsize=18)

    # 设置不同的Y轴刻度
    if model == 'DT':
        ax_ccfd.set_ylim(0.7, 1.02)
        ax_ccfd.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'RF':
        ax_ccfd.set_ylim(0.7, 1.02)
        ax_ccfd.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'XGBoost':
        ax_ccfd.set_ylim(0.7, 1.02)
        ax_ccfd.set_yticks([0.7, 0.8, 0.9, 1])
    elif model == 'LightGBM':
        ax_ccfd.set_ylim(0.3, 1.02)
        ax_ccfd.set_yticks([0.3, 0.5, 0.7, 0.9, 1])

    ax_ccfd.set_xticks(n_values)
    if (i // 2) + 2 == 3:  # 仅在底部图设置X轴标签
        ax_ccfd.set_xlabel('Number of Weak Classifiers (N)', fontsize=18)
    ax_ccfd.set_ylabel('Metric', fontsize=18)

    # 更改边框线为灰色
    ax_ccfd.spines['left'].set_color('gray')
    ax_ccfd.spines['left'].set_alpha(0.5)
    ax_ccfd.spines['bottom'].set_color('gray')
    ax_ccfd.spines['bottom'].set_alpha(0.5)

    ax_ccfd.spines['right'].set_visible(False)
    ax_ccfd.spines['top'].set_visible(False)

# 在图的底部显示一次图例
handles, labels = ax_sbd.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=18)

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.08)

# 保存图像为PDF
plt.savefig('sensitivity_analysis_sbd_ccfd.pdf', bbox_inches='tight')

plt.show()
