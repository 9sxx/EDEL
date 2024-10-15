import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Adult Income Dataset/adult.data"
path_test = '../../dataset/Adult Income Dataset/adult.test'

# 读取数据
# 给每一列加上属性名
columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'EdNum', 'MaritalStatus',
           'Occupation', 'Relationship', 'Race', 'Sex', 'CapitalGain',
           'CapitalLoss', 'HoursPerWeek', 'Country', 'Income']

data_train = pd.read_csv(path_train, names=columns)
data_test = pd.read_csv(path_test, names=columns, skiprows=1)

data = pd.concat([data_train, data_test], ignore_index=True)
print(data)

# 因为fnlgwt属性记录的是人口普查员的ID，对预测结果无影响，故删除该列
data.drop('fnlgwt', axis=1, inplace=True)
# 收入可以分为两种类型，则将'>50K'的替换成1（正样本），'<=50K'的替换成0（负样本）
data.replace({
    r'>50K': 1,
    r'<=50K': 0,
}, regex=True, inplace=True)
# 将数据集中‘ ?’字符替换为pd.NaT
data.replace(" ?", pd.NaT, inplace=True)
# 离散值的缺失值填充为众数
trans = {'Workclass': data['Workclass'].mode()[0], 'Occupation': data['Occupation'].mode()[0],
         'Country': data['Country'].mode()[0]}
data.fillna(trans, inplace=True)

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
print(data['Income'].value_counts())

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
