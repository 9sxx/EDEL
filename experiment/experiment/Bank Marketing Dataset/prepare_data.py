import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Bank Marketing Dataset/bank-additional-full.csv"

# 读取数据
data = pd.read_csv(path_train, sep=';')
print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 对data中的非数值列进行标签编码
for column in data.columns:
    if data[column].dtype == 'object':  # 检查列是否是非数值型
        # 初始化LabelEncoder
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 查看不平衡程度
print(data['y'].value_counts())

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
