import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Taiwan Credit Card Default Dataset/default of credit card clients.xls"

data = pd.read_excel(path_train, header=1)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

data.drop('ID', axis=1, inplace=True)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

print(data['default payment next month'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
