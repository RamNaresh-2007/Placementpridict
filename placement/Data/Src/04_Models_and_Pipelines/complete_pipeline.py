"""
M2: Supervised Learning — Linear Models at Depth

Topics Covered:
1. Linear regression revisited: closed-form normal equations, gradient-descent training, the comparison.
2. Regularisation: ridge (L2), lasso (L1), elastic net; the geometric picture of why L1 induces sparsity.
3. Logistic regression for binary classification: the sigmoid, the cross-entropy loss, the decision boundary.
4. Multinomial logistic regression (softmax regression) for multi-class problems.
5. Feature scaling and standardisation.
6. Categorical features: one-hot, ordinal, target, embedding-based (preview).
7. Handling missing values: deletion, mean/median, model-based, MissingIndicator.
8. Worked example end-to-end: ridge logistic regression to predict student placement.
"""

import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, os.path.join("..", "01_Datasets", os.path.join("..", "01_Datasets", "placement_predict_50k Dataset (2).csv")))

# -------------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------------
print("=== Loading Data ===")
df = pd.read_csv(FILE_PATH)
print(f"Data shape: {df.shape}")

# Introduce some missing values for demonstration
df_missing = df.copy()
np.random.seed(42)
mask = np.random.rand(len(df_missing)) < 0.1
df_missing.loc[mask, 'AptitudeTestScore'] = np.nan

# =========================================================================
# 7. Handling missing values: deletion, mean/median, model-based, MissingIndicator
# =========================================================================
print("\n=== Handling Missing Values ===")

# 7.1 Deletion
df_deleted = df_missing.dropna(subset=['AptitudeTestScore'])
print(f"Shape after deletion: {df_deleted.shape}")

# 7.2 Mean/Median Imputation & MissingIndicator
imputer = SimpleImputer(strategy='median', add_indicator=True)
imputed_array = imputer.fit_transform(df_missing[['AptitudeTestScore']])
print(f"Imputed with Median and MissingIndicator shape: {imputed_array.shape}")

# 7.3 Model-based imputation (KNN Imputer)
subset = df_missing[['AptitudeTestScore', 'CGPA']].head(1000)
knn_imputer = KNNImputer(n_neighbors=5)
subset_imputed = knn_imputer.fit_transform(subset)
print(f"KNN Imputed subset shape: {subset_imputed.shape}")

# =========================================================================
# 5. Feature scaling and standardisation
# =========================================================================
print("\n=== Feature Scaling and Standardisation ===")
print("When it matters: Gradient descent, distance-based models (KNN), regularization.")
print("When it doesn't: Tree-based models (Decision Trees, Random Forests, XGBoost).")

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

scaled_std = scaler_std.fit_transform(df[['CGPA', 'AptitudeTestScore']])
scaled_minmax = scaler_minmax.fit_transform(df[['CGPA', 'AptitudeTestScore']])

print(f"StandardScaler mean: {scaled_std.mean(axis=0)}, std: {scaled_std.std(axis=0)}")
print(f"MinMaxScaler min: {scaled_minmax.min(axis=0)}, max: {scaled_minmax.max(axis=0)}")

# =========================================================================
# 6. Categorical features: encoding methods
# =========================================================================
print("\n=== Categorical Features Encoding ===")

# 6.1 One-Hot Encoding (Nominal data)
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_city = ohe.fit_transform(df[['City']].head(100))
print(f"One-Hot Encoded City shape (first 100): {encoded_city.shape}")

# 6.2 Ordinal Encoding (Ordinal data)
ordinal = OrdinalEncoder()
encoded_hostel = ordinal.fit_transform(df[['Hostel']].head(100))
print(f"Ordinal Encoded Hostel shape: {encoded_hostel.shape}")

# 6.3 Target Encoding
print("Target Encoding: Replaces category with the mean of target for that category.")
target_mean = df.groupby('City')['PlacementStatus'].transform('mean')
print(f"Target Encoded City (first 5):\n{target_mean.head()}")

# 6.4 Embedding-based encoding (Preview)
print("Embedding-based encoding: Using neural networks to learn dense continuous vectors for categorical features.")

# =========================================================================
# 1. Linear regression revisited
# =========================================================================
print("\n=== Linear Regression: Normal Equations vs Gradient Descent ===")
X_reg = df[['CGPA']].values
y_reg = df['Salary Package'].values

X_b = np.c_[np.ones((len(X_reg), 1)), X_reg]

# 1.1 Closed-form Normal Equation: theta = (X^T X)^-1 X^T y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_reg)
print(f"Normal Equation Theta: {theta_best}")

# 1.2 Gradient-Descent Training
eta = 0.01
n_iterations = 1000
m = len(X_b)
np.random.seed(42)
theta_gd = np.random.randn(2)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y_reg)
    theta_gd = theta_gd - eta * gradients

print(f"Gradient Descent Theta: {theta_gd}")

# 1.3 Scikit-Learn comparison
lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
print(f"Scikit-Learn Theta: [{lin_reg.intercept_}, {lin_reg.coef_[0]}]")

# =========================================================================
# 2. Regularisation: Ridge (L2), Lasso (L1), Elastic Net
# =========================================================================
print("\n=== Regularisation (Ridge, Lasso, Elastic Net) ===")
print("L1 Sparsity Geometric Picture: L1 forms a diamond constraint in weight space. Loss contours hit corners -> exact zero weights.")

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
print(f"Ridge Coef: {ridge.coef_}, Lasso Coef: {lasso.coef_}, ElasticNet Coef: {elastic_net.coef_}")

# =========================================================================
# 3. Logistic Regression for Binary Classification
# =========================================================================
print("\n=== Logistic Regression for Binary Classification ===")
print("Sigmoid function: sigma(z) = 1 / (1 + exp(-z))")
print("Cross-entropy loss: Log loss for convex optimization.")
print("Decision Boundary: The hyperplane where theta^T x = 0.")

X_clf = df[['CGPA']]
y_clf = df['PlacementStatus']

log_reg = LogisticRegression()
log_reg.fit(X_clf, y_clf)
print(f"Binary LogReg Coef: {log_reg.coef_}, Intercept: {log_reg.intercept_}")

# =========================================================================
# 4. Multinomial Logistic Regression (Softmax Regression)
# =========================================================================
print("\n=== Multinomial Logistic Regression (Softmax Regression) ===")
X_multi = df[['CGPA', 'AptitudeTestScore']].copy()
y_multi = df['CollegeTier'].copy()

# Drop missing values to avoid errors
valid_idx = X_multi.dropna().index
X_multi = X_multi.loc[valid_idx]
y_multi = y_multi.loc[valid_idx]

softmax_reg = LogisticRegression(solver='lbfgs')
softmax_reg.fit(X_multi, y_multi)
print("Softmax Regression fitted for CollegeTier prediction.")

# =========================================================================
# 8. Worked example end-to-end: Ridge Logistic Regression
# =========================================================================
print("\n=== End-to-End Pipeline: Predict Placement with Ridge Logistic Regression ===")
features = ['CGPA', 'AttendancePercent', 'AptitudeTestScore', 'CodingTestScore', 'MockInterviewScore']
X_full = df[features]
y_full = df['PlacementStatus']

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('ridge_logreg', LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', random_state=42))
])

pipeline.fit(X_tr, y_tr)
y_pred = pipeline.predict(X_te)
print("Classification Report:")
print(classification_report(y_te, y_pred))

model = pipeline.named_steps['ridge_logreg']
print("Feature Coefficients:")
for feat, coef in zip(features, model.coef_[0]):
    print(f"  {feat}: {coef:.4f}")
