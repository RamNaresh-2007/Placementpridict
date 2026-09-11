# Multilinear regression
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
df = pd.read_csv(DATA_PATH)

# Features
X = df[
    [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]
]

# Target
y = df["Salary Package"]

# Remove missing values
data = pd.concat([X, y], axis=1).dropna()

X = data[
    [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]
]

y = data["Salary Package"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)

print("\nCoefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):
    print(feature, ":", coefficient)

print("\nModel Performance")
print("-----------------------")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# New student
new_student = pd.DataFrame([[
    8.5,    # CGPA
    80,     # Aptitude
    85,     # Coding
    75      # Interview
]], columns=["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"])

predicted_salary = model.predict(new_student)

print(
    "\nPredicted Salary:",
    predicted_salary[0],
    "LPA"
)