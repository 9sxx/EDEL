import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Adult Income Dataset/adult.data"
path_test = '../../dataset/Adult Income Dataset/adult.test'

columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'EdNum', 'MaritalStatus',
           'Occupation', 'Relationship', 'Race', 'Sex', 'CapitalGain',
           'CapitalLoss', 'HoursPerWeek', 'Country', 'Income']

data_train = pd.read_csv(path_train, names=columns)
data_test = pd.read_csv(path_test, names=columns, skiprows=1)

data = pd.concat([data_train, data_test], ignore_index=True)
print(data)

data.drop('fnlgwt', axis=1, inplace=True)
data.replace({
    r'>50K': 1,
    r'<=50K': 0,
}, regex=True, inplace=True)

data.replace(" ?", pd.NaT, inplace=True)

trans = {'Workclass': data['Workclass'].mode()[0], 'Occupation': data['Occupation'].mode()[0],
         'Country': data['Country'].mode()[0]}
data.fillna(trans, inplace=True)


for column in data.columns:
    if data[column].dtype == 'object':
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

print(data['Income'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
