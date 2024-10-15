import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from DEPSO import hybrid_DE_PSO_adjustment
from TLNN import TLNN_Search


def smote_tlnn_depso(Minority_data, Minority_label, Majority_data, Majority_label,
                     Synthetic_samples, Synthetic_label):
    data = np.concatenate((Minority_data, Majority_data), axis=0)
    t = np.concatenate((Minority_label, Majority_label), axis=0)

    improve_data = np.concatenate((data, Synthetic_samples), axis=0)
    improve_t = np.concatenate((t, Synthetic_label), axis=0)

    TLNN = TLNN_Search(improve_data)

    Noises = []
    for i in range(improve_data.shape[0]):
        labels = improve_t[TLNN[i]]
        pos = np.where(labels != improve_t[i])[0]
        if len(np.unique(pos)) >= 1 or len(TLNN[i]) == 0:
            Noises.append(i)

    Non_Noise = np.setdiff1d(np.arange(improve_t.size), Noises)
    Non_data = improve_data[Non_Noise, :]
    Non_t = improve_t[Non_Noise]
    Noises_data = improve_data[Noises, :]
    Noises_data_label = improve_t[Noises]

    if len(Noises) != 0:
        result = hybrid_DE_PSO_adjustment(Noises_data, Noises_data_label, Non_data, Non_t, 20)
        if result is None:
            GS, label_GS, Accuracy_global = np.array([]), np.array([]), 0.0
        else:
            GS, label_GS, Accuracy_global = result

    GS_reshaped = GS.reshape(GS.shape[0], Non_data.shape[1])
    Ip_data = np.concatenate((Non_data, GS_reshaped), axis=0)
    Ip_label = np.concatenate((Non_t, label_GS), axis=0)

    return Ip_data, Ip_label
