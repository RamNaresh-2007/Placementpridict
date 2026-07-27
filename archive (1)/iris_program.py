import pandas as pd

df = pd.read_csv("iris.csv")

print(df.head())
print(df.info())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())