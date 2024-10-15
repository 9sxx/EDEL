import matplotlib.pyplot as plt

n_values = [1, 2, 3, 4, 5]
results_aid = {
    'DT': {
        'AUC': [0.7717, 0.9506, 0.9519, 0.9531, 0.9514],
        'Accuracy': [0.8193, 0.9238, 0.9171, 0.9161, 0.9080],
        'Recall': [0.6085, 0.7489, 0.7907, 0.7703, 0.8042],
        'F1': [0.6171, 0.8194, 0.8215, 0.8110, 0.8087],
        'G-mean': [0.7341, 0.8526, 0.8685, 0.8580, 0.8686]
    },
    'RF': {
        'AUC': [0.8954, 0.9688, 0.9656, 0.9598, 0.9516],
        'Accuracy': [0.8513, 0.9280, 0.9148, 0.8998, 0.8748],
        'Recall': [0.6224, 0.7862, 0.8115, 0.7948, 0.8108],
        'F1': [0.6669, 0.8349, 0.8192, 0.7893, 0.7562],
        'G-mean': [0.7580, 0.8717, 0.8753, 0.8593, 0.8510]
    },
    'XGBoost': {
        'AUC': [0.9285, 0.9424, 0.9343, 0.9179, 0.9025],
        'Accuracy': [0.8728, 0.8828, 0.8621, 0.8377, 0.8164],
        'Recall': [0.6581, 0.6915, 0.7110, 0.6570, 0.6705],
        'F1': [0.7122, 0.7378, 0.7116, 0.6587, 0.6361],
        'G-mean': [0.7866, 0.8071, 0.8040, 0.7660, 0.7602]
    },
    'LightGBM': {
        'AUC': [0.9287, 0.9366, 0.9243, 0.9023, 0.8836],
        'Accuracy': [0.8741, 0.8770, 0.8509, 0.8172, 0.7898],
        'Recall': [0.6517, 0.6899, 0.6907, 0.6289, 0.6167],
        'F1': [0.7124, 0.7282, 0.6891, 0.6217, 0.5840],
        'G-mean': [0.7843, 0.8033, 0.7889, 0.7421, 0.7215]
    },
}

markers = {
    'AUC': 'o',
    'Accuracy': 's',
    'Recall': 'D',
    'F1': 'v',
    'G-mean': '^'
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams['axes.linewidth'] = 1.5

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for i, (model, metrics) in enumerate(results_aid.items()):
    ax = axs[i // 2, i % 2]

    for metric, values in metrics.items():
        ax.plot(n_values, values, label=metric, marker=markers[metric], markersize=6)

    ax.set_title(model, fontsize=14)
    ax.set_xlabel('Number of Weak Classifiers (N)', fontsize=12)
    ax.set_ylabel('Performance Metric', fontsize=12)
    ax.set_ylim(0.4, 1)
    ax.set_xticks(n_values)
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('aid_performance_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
