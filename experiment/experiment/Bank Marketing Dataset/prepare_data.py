import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Bank Marketing Dataset/bank-additional-full.csv"

data = pd.read_csv(path_train, sep=';')
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

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

print(data['y'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
