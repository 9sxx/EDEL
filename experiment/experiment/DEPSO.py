import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score


def sigmoid(x):
    """Sigmoid function."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def hybrid_DE_PSO_adjustment(GS, label_GS, TR, label_TR, Gm):
    # sort the solutions and labels
    label_GS_sorted = np.sort(label_GS)
    GS_sorted = GS[np.argsort(label_GS)]

    label_TR_sorted = np.sort(label_TR)
    TR_sorted = TR[np.argsort(label_TR)]

    G = 1
    X = GS_sorted
    label_X = label_GS_sorted
    X_n, D = X.shape
    V = np.zeros((X_n, D))

    X = X.astype(np.float64)  # 将 X 转换为 float64 类型
    V = V.astype(np.float64)  # 确保 V 也是 float64 类型

    max_col_X = np.max(X, axis=0)
    min_col_X = np.min(X, axis=0)
    class_n1 = len(np.unique(label_GS_sorted))
    class_n2 = len(np.unique(label_TR_sorted))
    class_n = max(class_n1, class_n2)

    # create an instance of the KNN classifier
    KNN_model = KNeighborsClassifier(n_neighbors=3).fit(TR_sorted, label_TR_sorted)
    Predict_label = KNN_model.predict(X)
    Accuracy_global = np.sum(label_X == Predict_label) / X.shape[0]

    print('-----------------Start accuracy (should be 0):', Accuracy_global, '------------------------')

    # 更新 correctIdx，不需要转换为列表
    correctIdx = np.zeros(len(label_GS_sorted))
    pos = np.where(label_X == Predict_label)[0]
    correctIdx[pos] = 1

    label_num_X, class_ni_X = np.unique(label_X, return_counts=True)
    label_num_TR, class_ni_TR = np.unique(label_TR, return_counts=True)

    while (Accuracy_global < 1 and G < Gm):
        if G > Gm:
            break

        Temp_TR = TR_sorted.copy()
        label_TR_Temp = label_TR_sorted.copy()
        Accuracy_U = 0

        print('--------------', G, ',', Accuracy_global, ',', Accuracy_U, '----------------------')

        for i in range(X_n):
            temporary = 0
            class_number_X = 1

            for j in range(len(class_ni_X)):
                temporary += class_ni_X[j]

                if i > temporary:
                    class_number_X += 1

            temporary = 0

            for j in range(class_number_X):
                temporary += class_ni_TR[j]

            # rand_r = temporary + np.random.permutation(class_ni_TR[class_number_X-1])
            # rand_ri = np.where(np.all(Temp_TR[rand_r] == X[i], axis=1))[0]
            rand_r = temporary + np.random.permutation(class_ni_TR[class_number_X - 1])[:4]
            # rand_ri = np.where(np.all(Temp_TR[rand_r] == X[i], axis=1))[0]
            rand_r[:min(4, len(X[i]))] = X[i][:min(4, len(X[i]))]

            # if len(rand_ri) > 0:
            #    rand_r = np.delete(rand_r, rand_ri[0])

            tau1 = 0.1
            tau2 = 0.1
            tau3 = 0.03
            tau4 = 0.07
            SFGSS = 8
            SFHC = 20
            Fl = 0.1
            Fu = 0.9
            Fi = np.random.rand(1)

            rand_Fi = np.random.rand(5)

            if rand_Fi[4] < tau3:
                Fi = SFGSS
            elif tau3 <= rand_Fi[4] and rand_Fi[4] < tau4:
                Fi = SFHC
            elif rand_Fi[1] < tau1:
                Fi = Fl + Fu * rand_Fi[0]

            KK = np.random.rand(1)
            # PSO update: Calculate the new velocity
            if (i < len(label_TR)):
                V[i] = np.multiply(sigmoid(V[i]), V[i]) \
                       + np.multiply(Fi * np.subtract(label_GS[i], X[i]), np.random.uniform()) \
                       + np.multiply(Fi * np.subtract(label_TR[i], X[i]), np.random.uniform())

            X[i] += V[i]

            # DE mutation
            if len(rand_r) >= 3:
                U = X[i, :] + KK * (TR[rand_r[0] % TR.shape[0], :] - X[i, :] + Fi * (
                        TR[rand_r[2] % TR.shape[0], :] - TR[rand_r[3] % TR.shape[0], :]))

            else:
                P_r1 = X[i, :] + X[i, :] * ((np.random.rand(D) - 0.5) / 5)
                P_r2 = X[i, :] + X[i, :] * ((np.random.rand(D) - 0.5) / 5)
                P_r3 = X[i, :] + X[i, :] * ((np.random.rand(D) - 0.5) / 5)
                U[i, :] = X[i, :] + KK * (P_r1 - X[i, :]) + Fi * (P_r2 - P_r3)

            Position_more1 = np.where(U > max_col_X)
            U[Position_more1] = max_col_X[Position_more1]
            Position_less0 = np.where(U < min_col_X)
            U[Position_less0] = min_col_X[Position_less0]

        # 找到需要更新的样本的索引
        incorrect_indices = np.where(correctIdx == 0)[0]

        for i in incorrect_indices:
            # 如果 U 是一维数组，则直接赋值
            if U.ndim == 1:
                X[i] = U
                GS[i] = U
            else:
                # 如果 U 是二维数组，则进行二维赋值
                X[i, :] = U[i, :]
                GS[i, :] = U[i, :]

        # create KNN classification model
        KNN_model = KNeighborsClassifier(n_neighbors=3)
        KNN_model.fit(TR, label_TR)
        # make predictions on new data
        Predict_label = KNN_model.predict(X)
        pos = np.where(label_X == Predict_label)
        Accuracy_U = sum(label_X == Predict_label) / X.shape[0];
        # assume that the true labels are stored in `label_X` and the predicted labels are stored in `Predict_label`
        f1 = f1_score(label_X, Predict_label, average='macro')
        correctIdx[pos] = 1;
        if (Accuracy_U > Accuracy_global):
            Accuracy_global = Accuracy_U
        print('--------------', G, ',', Accuracy_global, ',', Accuracy_U, '----------------------')
        print('--------------', G, ',', f1, '----------------------')
        G = G + 1
