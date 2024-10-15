import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/CDC Diabetes Health Indicators Dataset/diabetes_binary_health_indicators_BRFSS2015.csv"

# 读取数据
data = pd.read_csv(path_train)
print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 将 'Diabetes_binary' 列的数据类型转换为 int
data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
print(data)

# 查看不平衡程度
print(data['Diabetes_binary'].value_counts())

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
