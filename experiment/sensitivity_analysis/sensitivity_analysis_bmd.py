import matplotlib.pyplot as plt

n_values = [1, 2, 3, 4, 5]
results_bmd = {
    'DT': {
        'AUC': [0.7294, 0.9594, 0.9641, 0.9740, 0.9745],
        'Accuracy': [0.8892, 0.9657, 0.9639, 0.9658, 0.9632],
        'Recall': [0.5231, 0.7713, 0.8179, 0.7955, 0.8341],
        'F1': [0.5155, 0.8194, 0.8349, 0.8282, 0.8375],
        'G-mean': [0.6996, 0.8611, 0.8903, 0.8763, 0.8990]
    },
    'RF': {
        'AUC': [0.9444, 0.9881, 0.9865, 0.9842, 0.9795],
        'Accuracy': [0.9150, 0.9688, 0.9602, 0.9470, 0.9284],
        'Recall': [0.5168, 0.7841, 0.8179, 0.7948, 0.7994],
        'F1': [0.5781, 0.8344, 0.8185, 0.7656, 0.7145],
        'G-mean': [0.7064, 0.8715, 0.8894, 0.8708, 0.8665]
    },
    'XGBoost': {
        'AUC': [0.9458, 0.9756, 0.9761, 0.9745, 0.9708],
        'Accuracy': [0.9143, 0.9490, 0.9505, 0.9472, 0.9394],
        'Recall': [0.5485, 0.6466, 0.7291, 0.6996, 0.7420],
        'F1': [0.5904, 0.7330, 0.7679, 0.7419, 0.7333],
        'G-mean': [0.7258, 0.7930, 0.8418, 0.8213, 0.8428]
    },
    'LightGBM': {
        'AUC': [0.9508, 0.9687, 0.9679, 0.9632, 0.9550],
        'Accuracy': [0.9179, 0.9355, 0.9350, 0.9271, 0.9142],
        'Recall': [0.5636, 0.5584, 0.6472, 0.6235, 0.6772],
        'F1': [0.6074, 0.6569, 0.6904, 0.6541, 0.6403],
        'G-mean': [0.7366, 0.7376, 0.7908, 0.7724, 0.7986]
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

for i, (model, metrics) in enumerate(results_bmd.items()):
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
plt.savefig('bmd_performance_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
