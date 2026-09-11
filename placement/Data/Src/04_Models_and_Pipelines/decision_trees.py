"""
Decision Trees - Core Concepts and Implementation

Topics Covered:
1. Data Preparation for Tree Models (No scaling required).
2. Decision Tree Classifier: Predicting student placement (binary).
3. Decision Tree Regressor: Predicting salary package (continuous).
4. Preventing Overfitting: Pruning via hyperparameters (max_depth, min_samples_split).
5. Interpretability: Feature Importances and Tree structure extraction.
"""

import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, os.path.join("..", "01_Datasets", os.path.join("..", "01_Datasets", "placement_sample_20.csv")))

print("=== Loading Data ===")
df = pd.read_csv(FILE_PATH)
print(f"Data shape: {df.shape}")

# Features and Targets
features = ['CGPA', 'AttendancePercent', 'AptitudeTestScore', 'CodingTestScore', 'MockInterviewScore']
X = df[features]
y_clf = df['PlacementStatus']  # Classification target
y_reg = df['Salary Package']   # Regression target

# Split the dataset
X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)

# Imputer for missing values (Tree models in sklearn cannot handle NaNs natively yet)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# =========================================================================
# 1. Decision Tree Classifier
# =========================================================================
print("\n=== 1. Decision Tree Classifier (Predicting PlacementStatus) ===")
print("Concept: Splits data at nodes to maximize Information Gain / minimize Gini Impurity.")

# Unpruned Tree (can overfit easily)
clf_unpruned = DecisionTreeClassifier(random_state=42)
clf_unpruned.fit(X_train_imputed, y_clf_train)
print(f"Unpruned Tree Depth: {clf_unpruned.get_depth()}")
print(f"Unpruned Tree Leaves: {clf_unpruned.get_n_leaves()}")
print(f"Train Accuracy: {clf_unpruned.score(X_train_imputed, y_clf_train):.4f}")
print(f"Test Accuracy: {clf_unpruned.score(X_test_imputed, y_clf_test):.4f}")

# Pruned Tree (using hyperparameters to generalize better)
print("\n--- Applying Pruning (max_depth=3, min_samples_leaf=2) ---")
clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
clf_pruned.fit(X_train_imputed, y_clf_train)

y_clf_pred = clf_pruned.predict(X_test_imputed)
print("\nPruned Tree Classification Report:")
print(classification_report(y_clf_test, y_clf_pred))

# =========================================================================
# 2. Interpretability: Feature Importance and Tree Rules
# =========================================================================
print("\n=== 2. Feature Importances ===")
importances = clf_pruned.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

print("\n--- Extracting Tree Rules (Top 3 levels) ---")
tree_rules = export_text(clf_pruned, feature_names=features, max_depth=3)
print(tree_rules)

# =========================================================================
# 3. Decision Tree Regressor
# =========================================================================
print("\n=== 3. Decision Tree Regressor (Predicting Salary Package) ===")
print("Concept: Minimizes Mean Squared Error (MSE) at each split for continuous values.")

# Drop rows where Salary Package is missing just in case
reg_mask = y_reg_train.notna()
X_reg_train = X_train_imputed[reg_mask]
y_reg_train_clean = y_reg_train[reg_mask]

reg_mask_test = y_reg_test.notna()
X_reg_test = X_test_imputed[reg_mask_test]
y_reg_test_clean = y_reg_test[reg_mask_test]

# Train Regressor
reg = DecisionTreeRegressor(max_depth=6, random_state=42)
reg.fit(X_reg_train, y_reg_train_clean)

y_reg_pred = reg.predict(X_reg_test)
mse = mean_squared_error(y_reg_test_clean, y_reg_pred)
r2 = r2_score(y_reg_test_clean, y_reg_pred)

print(f"Regressor Mean Squared Error: {mse:,.2f}")
print(f"Regressor R^2 Score: {r2:.4f}")
