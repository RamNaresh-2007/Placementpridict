# Data Leakage Prevention & Robust Model Evaluation - Session 28

## 1. Overview & Theoretical Principles

In machine learning, **Data Leakage** (or information leakage) is an insidious failure mode where information from outside the training partition—such as the validation fold, unseen holdout test set, or future timeline—is inadvertently incorporated into model training or feature preprocessing.

When data leakage occurs:
1. **The Optimism Bias**: Cross-validation and test scores appear exceptionally strong during offline research and development.
2. **Production Crash**: Once deployed into production on genuinely independent student cohorts, the model's performance drastically collapses because the leaked information (global means, variance, target distributions, or snooped hyperparameter choices) does not exist in the real world.

This module addresses and implements the **two gold-standard solutions** to data leakage:
- **Solution A: Vault-Lock Stratification & Pipeline Encapsulation**
- **Solution B: Nested Cross-Validation (Double CV) for Hyperparameter Optimization**

---

## 2. The Leaky Workflow vs. The Leakage-Free Workflow

### The Flawed Leaky Pattern (Naive Machine Learning)
```python
# FATAL FLAW: Scaling / Imputing on the entire dataset BEFORE splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # <-- LEAKAGE! Computes μ and σ using ALL rows

# The validation set influenced the scaler!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20)
model = LogisticRegression()
model.fit(X_train, y_train)
```
- **Why this fails**: The global mean $\mu_{global} = \frac{1}{N} \sum_{i=1}^N x_i$ and standard deviation $\sigma_{global}$ contain information from the test set. When test instances are evaluated, the model evaluates points scaled by their own distribution!

---

### Implementation 1: Vault-Lock & Encapsulated Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Load Placement Dataset
X = df[['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating', 'Internships', 'Projects']]
y = df['PlacementStatus']

# 2. Vault-Lock 20% Test Set with Stratification
# The test set is isolated immediately and locked away in a "vault"
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create Leakage-Free Pipeline (Scaler + Estimator)
# Inside the pipeline, scaler.fit() only runs on training folds!
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# 4. Perform 4-Fold Stratified Cross-Validation on Development Set
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring='accuracy')
print(f'Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')

# 5. Retrain on Full Dev Set & Evaluate Unseen Test Set (Once and only once)
pipeline.fit(X_dev, y_dev)
print(f'Test Accuracy: {pipeline.score(X_test, y_test):.4f}')
```

---

### Implementation 2: Nested Cross-Validation (Double CV)

When tuning hyperparameters (e.g. $C$ in regularized Logistic Regression), traditional `GridSearchCV.best_score_` is **optimistically biased** because the hyperparameters were selected to maximize the validation performance on those specific folds ("Hyperparameter Snooping").

**Nested Cross-Validation** eliminates this tuning leakage by splitting the procedure into two concentric loops:
- **Outer Loop (5-Fold Stratified CV)**: Estimates unbiased generalization performance.
- **Inner Loop (3-Fold Stratified CV)**: Tunes hyperparameters on outer training data only.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Load Placement Dataset
X = df[['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating', 'Internships', 'Projects']]
y = df['PlacementStatus']

# 2. Define Leakage-Free Pipeline (Scaler + Estimator)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# 3. Define Outer (5-Fold) & Inner (3-Fold) Stratified Schemes
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 4. Inner Loop Grid Search over Hyperparameter Space C
param_grid = {'model__C': [0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1')

# 5. Outer Loop Unbiased Performance Estimation
nested_f1_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1', n_jobs=-1)

print(f"Outer Fold F1-Scores: {nested_f1_scores.round(4)}")
print(f"Unbiased Nested CV F1-Score: {nested_f1_scores.mean():.4f} (+/- {nested_f1_scores.std()*2:.4f})")
```

---

## 3. Directory Structure

```
data_leakage/
├── data_leakage_pipeline.py                 # Core pipeline, benchmarks & chart generation
├── leakage_report.txt                        # Formatted textual analytical report
├── leakage_summary.json                      # Structured JSON telemetry for web APIs
├── 01_leakage_concept_architecture.png       # Conceptual flow diagram comparing 3 paradigms
├── 02_leaky_vs_leakage_free_benchmark.png    # Bar chart quantifying Optimism Gap
├── 03_distribution_contamination_proof.png   # Proof of feature drift (global vs fold scaler)
├── 04_nested_cv_outer_vs_inner_performance.png # Outer fold F1 scores & tuning stability
├── 05_vault_lock_test_evaluation.png         # Dev CV vs Vault Test, Confusion Matrix, ROC
├── plots/                                    # Mirror directory containing high-res figures
└── README.md                                 # Technical documentation and guide
```

---

## 4. Visual Analytics Generated

1. **`01_leakage_concept_architecture.png`**:
   - Panel A: Naive Leaky Pipeline illustrating global contamination.
   - Panel B: Vault-Lock & Pipeline Workflow (Image 2).
   - Panel C: Nested Cross-Validation Outer/Inner loops (Image 1).
2. **`02_leaky_vs_leakage_free_benchmark.png`**:
   - Side-by-side comparison across Accuracy, Precision, Recall, F1, and ROC-AUC with explicit calculation of the Optimism Bias ($\Delta = \text{Score}_{leaky} - \text{Score}_{clean}$).
3. **`03_distribution_contamination_proof.png`**:
   - Compares $\mu$ and $\sigma$ calculated globally vs. strictly on training folds for `CGPA`, `AptitudeTestScore`, `CodingTestScore`, and `SoftSkillsRating`.
4. **`04_nested_cv_outer_vs_inner_performance.png`**:
   - Visualizes fold-by-fold generalization scores, 95% confidence intervals ($\pm 2\sigma$), and compares inner CV tuned scores against unseen outer validation folds.
5. **`05_vault_lock_test_evaluation.png`**:
   - 3-panel dashboard depicting Development 4-Fold CV consistency against the Vault Test set, alongside the confusion matrix and ROC curve (AUC).

---

## 5. Execution Instructions

Run from the project root:
```powershell
.\.venv\Scripts\python "data_leakage.py"
```

Or from the session runner:
```powershell
.\.venv\Scripts\python "placement/Data/Src/04_Models_and_Pipelines/28_data_leakage.py"
```

Or run the pipeline directly:
```powershell
.\.venv\Scripts\python "placement/Data/Src/04_Models_and_Pipelines/data_leakage/data_leakage_pipeline.py"
```
