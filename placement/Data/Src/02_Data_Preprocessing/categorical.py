import pandas as pd

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv"))

print("========== GENDER ==========")
print(df["Gender"].head(10))

print("\n========== CITY ==========")
print(df["City"].head(10))

print("\n========== COLLEGE TIER ==========")
print(df["CollegeTier"].head(10))

print("\n========== STREAM ==========")
print(df["Stream"].head(10))

print("\n========== SPECIALISATION ==========")
print(df["Specialisation"].head(10))

print("\n========== HOSTEL ==========")
print(df["Hostel"].head(10))

print("\n========== HISTORY OF BACKLOGS ==========")
print(df["HistoryOfBacklogs"].head(10))

print("\n========== CGPA TIER ==========")
print(df["CGPA_Tier"].head(10))