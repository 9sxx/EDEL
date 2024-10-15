import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Credit Card Fraud Detection/creditcard.csv"

data = pd.read_csv(path_train)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

data.drop('Time', axis=1, inplace=True)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

print(data['Class'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
