import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Taiwan Credit Card Default Dataset/default of credit card clients.xls"

# 读取数据
data = pd.read_excel(path_train, header=1)
print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

data.drop('ID', axis=1, inplace=True)
print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 查看不平衡程度
print(data['default payment next month'].value_counts())

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
