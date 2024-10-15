import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/Give Me Some Credit Dataset/cs-training.csv"

data = pd.read_csv(path_train)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

for col in data.columns:
    if data[col].isnull().any():
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

print(data['SeriousDlqin2yrs'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
