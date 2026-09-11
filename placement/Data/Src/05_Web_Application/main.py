import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.info())