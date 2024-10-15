import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../dataset/CDC Diabetes Health Indicators Dataset/diabetes_binary_health_indicators_BRFSS2015.csv"

data = pd.read_csv(path_train)
print(data)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)
print(data)

print(data['Diabetes_binary'].value_counts())

data.to_csv("data.csv", index=False)

print("Data processing successful!")
