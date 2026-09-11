# Logistic Regression - sigmoid, cross-entropy, decision boundary, softmax
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "05_Web_Application"))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import config

os.makedirs(config.LOGISTIC_DIR, exist_ok=True)

# ---------- Load data and prepare features (numeric columns only) ----------
from sklearn.model_selection import train_test_split
df = pd.read_csv(config.RAW_DATA_PATH)
train, val = train_test_split(df, test_size=0.2, random_state=config.RANDOM_STATE)

feature_cols = config.NUMERIC_COLUMNS
x_train, y_train = train[feature_cols].copy(), train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols].copy(), val[config.TARGET_COLUMN]

# fill missing values with the training median, same rule as before
for col in feature_cols:
    fill_value = x_train[col].median()
    x_train[col] = x_train[col].fillna(fill_value)
    x_val[col] = x_val[col].fillna(fill_value)

# ---------- Class balance: how many students are Placed vs Not Placed ----------

placed_pct = (y_train == 1).mean() * 100
print(f"Placed: {round(placed_pct,1)}%  |  Not Placed: {round(100-placed_pct,1)}%")

# ---------- CGPA vs Placement: the empirical S-curve ----------

bins = pd.cut(train["CGPA"], bins=15)
fraction_placed = train.groupby(bins, observed=True)[config.TARGET_COLUMN].mean()
bin_centers = [interval.mid for interval in fraction_placed.index]

plt.figure(figsize=(7, 5))
plt.plot(bin_centers, fraction_placed.values, marker="o")
plt.xlabel("CGPA")
plt.ylabel("Fraction Placed")
plt.title("CGPA vs Placement - the S-shaped pattern")
plt.savefig(f"{config.LOGISTIC_DIR}/cgpa_vs_placement_curve.png")
plt.close()
print("CGPA vs placement S-curve saved")

# ---------- Which single feature predicts placement best? (AUC) ----------

auc_features = ["CGPA", "MockInterviewScore", "CodingTestScore", "AptitudeTestScore", "AttendancePercent"]
print("\nSingle-feature AUC (1.0 = perfect, 0.5 = no better than chance):")
auc_scores = {}
for col in auc_features:
    auc = roc_auc_score(y_train, x_train[col])
    auc_scores[col] = auc
    print(f"{col:<20}: {round(auc, 4)}")

# ---------- Why not a straight line? ----------

linreg = LinearRegression()
linreg.fit(x_train[["CGPA"]], y_train)
line_predictions = linreg.predict(x_train[["CGPA"]])
print(f"\nLinear regression on CGPA -> PlacementStatus gives predictions from "
      f"{round(line_predictions.min(),2)} to {round(line_predictions.max(),2)} "
      f"(a probability must stay between 0 and 1 - this is why we need logistic regression)")

# ---------- The sigmoid function ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-8, 8, 100)
plt.figure(figsize=(7, 5))
plt.plot(z_values, sigmoid(z_values))
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("The Sigmoid Function - always between 0 and 1")
plt.savefig(f"{config.LOGISTIC_DIR}/sigmoid_function.png")
plt.close()
print("Sigmoid function plot saved")

# ---------- Fit the full logistic regression model ----------

model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
model.fit(x_train, y_train)

val_accuracy = accuracy_score(y_val, model.predict(x_val))
val_log_loss = log_loss(y_val, model.predict_proba(x_val))
print(f"\nFull model - Validation accuracy: {round(val_accuracy, 4)}")
print(f"Full model - Cross-entropy (log loss): {round(val_log_loss, 4)}")

# ---------- Decision boundary using 2 features ----------

boundary_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
boundary_model.fit(x_train[["CGPA", "CodingTestScore"]].to_numpy(), y_train)

x_min, x_max = x_train["CGPA"].min() - 0.5, x_train["CGPA"].max() + 0.5
y_min, y_max = x_train["CodingTestScore"].min() - 5, x_train["CodingTestScore"].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_predictions = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, grid_predictions, alpha=0.2, levels=1, colors=["#f4a6a6", "#a6c8f4"])
plt.scatter(x_val["CGPA"], x_val["CodingTestScore"], c=y_val, cmap="coolwarm", alpha=0.3, s=10)
plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("Decision Boundary (CGPA + Coding Test Score)")
plt.savefig(f"{config.LOGISTIC_DIR}/decision_boundary.png")
plt.close()
print("Decision boundary plot saved")

# ---------- Scaler comparison: Unscaled vs StandardScaler vs MinMaxScaler ----------

def evaluate_logistic_regression(x_tr, x_va, y_tr, y_va, label):
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    model.fit(x_tr, y_tr)
    acc = accuracy_score(y_va, model.predict(x_va))
    print(f"{label:<20}: Validation Accuracy = {round(acc, 4)}")
    return acc

print("\nScaler comparison (same model, three different inputs):")
results = {}
results["Unscaled"] = evaluate_logistic_regression(x_train, x_val, y_train, y_val, "Unscaled")

standard_scaler = StandardScaler()
x_train_std = standard_scaler.fit_transform(x_train)
x_val_std = standard_scaler.transform(x_val)
results["StandardScaler"] = evaluate_logistic_regression(x_train_std, x_val_std, y_train, y_val, "StandardScaler")

minmax_scaler = MinMaxScaler()
x_train_mm = minmax_scaler.fit_transform(x_train)
x_val_mm = minmax_scaler.transform(x_val)
results["MinMaxScaler"] = evaluate_logistic_regression(x_train_mm, x_val_mm, y_train, y_val, "MinMaxScaler")

# ---------- Bonus: multinomial logistic regression (softmax), 3 classes ----------

salary_median = train.loc[train["Salary Package"] > 0, "Salary Package"].median()

def make_package_tier(salary):
    if salary == 0:
        return "Not Placed"
    elif salary < salary_median:
        return "Standard Package"
    return "Premium Package"

y_train_tier = train["Salary Package"].apply(make_package_tier)
y_val_tier = val["Salary Package"].apply(make_package_tier)

tier_scaler = StandardScaler()
x_train_tier = tier_scaler.fit_transform(x_train)
x_val_tier = tier_scaler.transform(x_val)

softmax_model = LogisticRegression(solver="lbfgs", max_iter=1000)
softmax_model.fit(x_train_tier, y_train_tier)
tier_accuracy = accuracy_score(y_val_tier, softmax_model.predict(x_val_tier))
print(f"\nSoftmax (3-class: Not Placed / Standard / Premium) accuracy: {round(tier_accuracy, 4)}")

# ---------- Save a report with all key numbers ----------

with open(f"{config.LOGISTIC_DIR}/logistic_regression_report.txt", "w") as f:
    f.write(f"Placed: {round(placed_pct,1)}%  Not Placed: {round(100-placed_pct,1)}%\n\n")
    f.write("Single-feature AUC:\n")
    for col, auc in auc_scores.items():
        f.write(f"  {col}: {round(auc,4)}\n")
    f.write(f"\nFull model validation accuracy: {round(val_accuracy,4)}\n")
    f.write(f"Full model cross-entropy (log loss): {round(val_log_loss,4)}\n")
    f.write("\nScaler comparison:\n")
    for label, acc in results.items():
        f.write(f"  {label}: {round(acc,4)}\n")
    f.write(f"\nSoftmax 3-class accuracy: {round(tier_accuracy,4)}\n")
