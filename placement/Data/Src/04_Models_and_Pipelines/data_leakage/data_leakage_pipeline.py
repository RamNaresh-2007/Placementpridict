# ==============================================================================
# Data Leakage Prevention & Robust Model Evaluation - Session 28
# ==============================================================================
# Core Engineering Pipeline implementing:
# 1. Leakage-Free Pipeline & Nested Cross-Validation (Outer 5-Fold, Inner 3-Fold)
# 2. Vault-Lock Stratified Test Partitioning & Development Set Evaluation
# 3. Side-by-Side Leaky vs. Leak-Free Empirical Benchmark (Quantifying Optimism Bias)
# 4. Mathematical Distribution Contamination Proof (Global vs Fold Scaler Shift)
# 5. Publication-grade Visual Analytics Suite
# ==============================================================================

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Robust path handling across execution environments
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(models_dir)
app_dir = os.path.join(src_dir, "05_Web_Application")

for path_dir in [app_dir, src_dir, models_dir, current_dir]:
    if os.path.exists(path_dir) and path_dir not in sys.path:
        sys.path.insert(0, path_dir)

try:
    import config
except ImportError:
    config = None


def setup_directories():
    """Create and return all target directories for artifacts and visualizations."""
    base_leak_dir = current_dir
    plots_dir = os.path.join(base_leak_dir, "plots")
    
    if config and hasattr(config, "DATA_LEAKAGE_OUTPUTS_DIR"):
        outputs_dir = config.DATA_LEAKAGE_OUTPUTS_DIR
    else:
        outputs_dir = os.path.join(src_dir, "06_Outputs_and_Utils", "outputs", "data_leakage")
        
    web_plots_dir = os.path.join(app_dir, "static", "plots", "data_leakage")
    
    target_dirs = [base_leak_dir, plots_dir, outputs_dir, web_plots_dir]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        
    return {
        "base": base_leak_dir,
        "plots": plots_dir,
        "outputs": outputs_dir,
        "web": web_plots_dir
    }


def load_dataset(sample_size=None, random_state=42):
    """
    Load student placement dataset with the exact features from user specification:
    X: ['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating', 'Internships', 'Projects']
    y: 'PlacementStatus'
    """
    cleaned_path = os.path.join(src_dir, "06_Outputs_and_Utils", "processed", "cleaned_data.csv")
    raw_path = os.path.join(src_dir, "01_Datasets", "placement_predict_50k Dataset (2).csv")
    
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    elif config and hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
        df = pd.read_csv(config.RAW_DATA_PATH)
    else:
        raise FileNotFoundError("Could not locate placement dataset.")

    features = ['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating', 'Internships', 'Projects']
    target = 'PlacementStatus'

    # Clean missing values if any
    df_clean = df.dropna(subset=features + [target]).copy()

    # Subsample if specified for rapid responsive CV execution
    if sample_size and len(df_clean) > sample_size:
        df_clean = df_clean.sample(n=sample_size, random_state=random_state)

    X = df_clean[features].copy()
    y = df_clean[target].astype(int).copy()

    return X, y, df_clean


def save_figure(fig, filename, dirs):
    """Save matplotlib figure to all configured asset destinations in high-res."""
    targets = [
        os.path.join(dirs["base"], filename),
        os.path.join(dirs["plots"], filename),
        os.path.join(dirs["outputs"], filename),
        os.path.join(dirs["web"], filename)
    ]
    for target in targets:
        fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# WORKFLOW 1: VAULT-LOCK STRATIFIED HOLDOUT SET & DEV PIPELINE (IMAGE 2)
# ==============================================================================
def run_vault_lock_workflow(X, y, dirs, random_state=42):
    """
    Implementation of User Image 2:
    1. Vault-Lock 20% Test Set with Stratification
    2. Create Leakage-Free Pipeline (StandardScaler + LogisticRegression)
    3. Perform 4-Fold Stratified Cross-Validation on Development Set
    4. Retrain on Full Dev Set & Evaluate Unseen Test Set
    """
    print("\n" + "="*70)
    print(">>> WORKFLOW 1: VAULT-LOCK TEST SET & STRATIFIED DEV CV (IMAGE 2)")
    print("="*70)

    # 1. Vault-Lock 20% Test Set with Stratification
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"[Vault-Lock] Total samples: {len(X):,}")
    print(f"[Vault-Lock] Development Set (80%): {len(X_dev):,} | Vault Test Set (20%): {len(X_test):,}")
    print(f"[Vault-Lock] Dev Class Distribution: {dict(y_dev.value_counts(normalize=True).round(3))}")
    print(f"[Vault-Lock] Test Class Distribution: {dict(y_test.value_counts(normalize=True).round(3))}")

    # 2. Create Leakage-Free Pipeline (Scaler + Estimator)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))

    # 3. Perform 4-Fold Stratified Cross-Validation on Development Set
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring='f1')
    cv_auc = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring='roc_auc')

    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Mean CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
    print(f"Mean CV ROC-AUC:  {cv_auc.mean():.4f} (+/- {cv_auc.std()*2:.4f})")

    # 4. Retrain on Full Dev Set & Evaluate Unseen Test Set
    pipeline.fit(X_dev, y_dev)
    test_acc = pipeline.score(X_test, y_test)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC-AUC:  {test_auc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    results = {
        "dev_cv_scores": cv_scores.tolist(),
        "dev_cv_mean": float(cv_scores.mean()),
        "dev_cv_std": float(cv_scores.std()),
        "dev_cv_f1_mean": float(cv_f1.mean()),
        "dev_cv_auc_mean": float(cv_auc.mean()),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_auc": float(test_auc),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

    return results, pipeline, (X_dev, X_test, y_dev, y_test)


# ==============================================================================
# WORKFLOW 2: NESTED CROSS-VALIDATION (IMAGE 1)
# ==============================================================================
def run_nested_cv_workflow(X, y, dirs, random_state=42):
    """
    Implementation of User Image 1:
    1. Define Leakage-Free Pipeline (Scaler + Estimator)
    2. Outer (5-Fold) & Inner (3-Fold) Stratified Schemes
    3. Inner Loop Grid Search over Hyperparameter Space C
    4. Outer Loop Unbiased Performance Estimation (Nested CV)
    """
    print("\n" + "="*70)
    print(">>> WORKFLOW 2: NESTED CROSS-VALIDATION & GRID SEARCH (IMAGE 1)")
    print("="*70)

    # 1. Define Leakage-Free Pipeline (Scaler + Estimator)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # 2. Define Outer (5-Fold) & Inner (3-Fold) Stratified Schemes
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # 3. Inner Loop Grid Search over Hyperparameter Space C
    param_grid = {'model__C': [0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1')

    # 4. Outer Loop Unbiased Performance Estimation
    print("[Nested CV] Executing 5-Fold Outer Loop with 3-Fold Inner Grid Search...")
    t0 = time.time()
    nested_f1_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1', n_jobs=-1)
    nested_acc_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy', n_jobs=-1)
    nested_auc_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)
    elapsed = time.time() - t0

    print(f"Outer Fold F1-Scores: {nested_f1_scores.round(4)}")
    print(f"Unbiased Nested CV F1-Score: {nested_f1_scores.mean():.4f} (+/- {nested_f1_scores.std()*2:.4f})")
    print(f"Outer Fold Accuracies: {nested_acc_scores.round(4)}")
    print(f"Unbiased Nested CV Accuracy: {nested_acc_scores.mean():.4f} (+/- {nested_acc_scores.std()*2:.4f})")
    print(f"Outer Fold ROC-AUCs: {nested_auc_scores.round(4)}")
    print(f"Unbiased Nested CV ROC-AUC:  {nested_auc_scores.mean():.4f} (+/- {nested_auc_scores.std()*2:.4f})")
    print(f"[Nested CV] Completed in {elapsed:.2f}s")

    # Detailed fold inspection: extract best parameters per outer fold
    outer_fold_details = []
    fold_idx = 1
    for train_ix, val_ix in outer_cv.split(X, y):
        X_tr, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_tr, y_val = y.iloc[train_ix], y.iloc[val_ix]

        gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1')
        gs.fit(X_tr, y_tr)
        val_pred = gs.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)

        outer_fold_details.append({
            "fold": fold_idx,
            "best_C": gs.best_params_['model__C'],
            "inner_cv_best_f1": float(gs.best_score_),
            "outer_val_f1": float(val_f1),
            "outer_val_acc": float(val_acc)
        })
        fold_idx += 1

    # Standard non-nested grid search on whole dataset (optimistic tuning estimate)
    gs_naive = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=outer_cv, scoring='f1')
    gs_naive.fit(X, y)
    naive_best_f1 = float(gs_naive.best_score_)
    print(f"[Tuning Leakage Proof] Non-nested best CV score: {naive_best_f1:.4f} vs Unbiased Nested CV: {nested_f1_scores.mean():.4f}")

    results = {
        "outer_f1_scores": nested_f1_scores.tolist(),
        "nested_f1_mean": float(nested_f1_scores.mean()),
        "nested_f1_std": float(nested_f1_scores.std()),
        "outer_acc_scores": nested_acc_scores.tolist(),
        "nested_acc_mean": float(nested_acc_scores.mean()),
        "nested_acc_std": float(nested_acc_scores.std()),
        "outer_auc_scores": nested_auc_scores.tolist(),
        "nested_auc_mean": float(nested_auc_scores.mean()),
        "nested_auc_std": float(nested_auc_scores.std()),
        "non_nested_optimistic_f1": naive_best_f1,
        "tuning_optimism_delta": float(naive_best_f1 - nested_f1_scores.mean()),
        "fold_details": outer_fold_details
    }

    return results


# ==============================================================================
# WORKFLOW 3: LEAKY VS. LEAKAGE-FREE EMPIRICAL BENCHMARK
# ==============================================================================
def run_leakage_comparison_experiment(X, y, random_state=42):
    """
    Direct empirical comparison proving the statistical impact of data leakage:
    - Path A (Leaky): Pre-scaling the entire dataset BEFORE CV splits (global mean & variance leak)
    - Path B (Leakage-Free): Strict Pipeline encapsulation with fold-isolated scaling
    """
    print("\n" + "="*70)
    print(">>> WORKFLOW 3: LEAKY VS. LEAKAGE-FREE EMPIRICAL BENCHMARK")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # --- PATH A: LEAKY PREPROCESSING ---
    # Global scaling before CV partition
    scaler_global = StandardScaler()
    X_leaky = pd.DataFrame(scaler_global.fit_transform(X), columns=X.columns, index=X.index)

    model_leaky = LogisticRegression(max_iter=1000, random_state=random_state)
    cv_leaky = cross_validate(model_leaky, X_leaky, y, cv=cv, scoring=scoring, return_train_score=True)

    # --- PATH B: STRICT LEAKAGE-FREE PIPELINE ---
    pipeline_clean = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    cv_clean = cross_validate(pipeline_clean, X, y, cv=cv, scoring=scoring, return_train_score=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    comparison = {}

    print(f"{'Metric':<15} | {'Leaky (Contaminated)':<22} | {'Leakage-Free (Pipeline)':<24} | {'Optimism Gap':<15}")
    print("-" * 85)

    for m in metrics:
        leaky_m = float(cv_leaky[f'test_{m}'].mean())
        clean_m = float(cv_clean[f'test_{m}'].mean())
        gap = leaky_m - clean_m
        comparison[m] = {
            "leaky_mean": leaky_m,
            "leaky_std": float(cv_leaky[f'test_{m}'].std()),
            "clean_mean": clean_m,
            "clean_std": float(cv_clean[f'test_{m}'].std()),
            "gap": gap,
            "leaky_folds": cv_leaky[f'test_{m}'].tolist(),
            "clean_folds": cv_clean[f'test_{m}'].tolist()
        }
        print(f"{m.capitalize():<15} | {leaky_m:.4f} (+/- {comparison[m]['leaky_std']*2:.4f})  | "
              f"{clean_m:.4f} (+/- {comparison[m]['clean_std']*2:.4f})    | {gap:+.4f}")

    return comparison


# ==============================================================================
# WORKFLOW 4: MATHEMATICAL DISTRIBUTION CONTAMINATION PROOF
# ==============================================================================
def analyze_distribution_contamination(X, y, random_state=42):
    """
    Quantifies the mathematical difference in mean (mu) and variance (sigma)
    between fitting globally on entire dataset vs. isolated training folds.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(skf.split(X, y))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    global_scaler = StandardScaler().fit(X)
    train_scaler = StandardScaler().fit(X_train)

    feature_contamination = []
    for i, col in enumerate(X.columns):
        mu_global = global_scaler.mean_[i]
        mu_train = train_scaler.mean_[i]
        sig_global = np.sqrt(global_scaler.var_[i])
        sig_train = np.sqrt(train_scaler.var_[i])

        # Standardized difference for val points
        z_global = (X_val[col] - mu_global) / sig_global
        z_clean = (X_val[col] - mu_train) / sig_train
        max_shift = np.max(np.abs(z_global - z_clean))
        mean_shift = np.mean(np.abs(z_global - z_clean))

        feature_contamination.append({
            "feature": col,
            "mu_global": float(mu_global),
            "mu_train": float(mu_train),
            "mu_shift": float(abs(mu_global - mu_train)),
            "sig_global": float(sig_global),
            "sig_train": float(sig_train),
            "sig_shift": float(abs(sig_global - sig_train)),
            "mean_z_distortion": float(mean_shift),
            "max_z_distortion": float(max_shift)
        })

    return feature_contamination, (X_val, global_scaler, train_scaler)


# ==============================================================================
# VISUALIZATION SUITE (PUBLICATION-GRADE PLOTS)
# ==============================================================================
def generate_all_plots(vault_results, nested_results, comparison_results, contamination_data, dirs):
    """Generate and save 5 publication-grade visualization charts."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette_primary = ["#1E3A8A", "#0D9488", "#F59E0B", "#EF4444", "#8B5CF6"]

    # --------------------------------------------------------------------------
    # PLOT 1: CONCEPTUAL ARCHITECTURE & DATA FLOW (LEAKY VS VAULT VS NESTED)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 8.2))
    fig.suptitle("Data Leakage Taxonomy & Leakage-Free Validation Paradigms", fontsize=15, fontweight="bold", y=0.98)

    # Subplot A: The Flawed Leaky Path
    ax = axes[0]
    ax.axis("off")
    ax.set_title("A. Flawed Leaky Pipeline (Global Preprocessing -> Contaminated CV Splits)", fontsize=12, fontweight="bold", loc="left", color="#DC2626")
    ax.text(0.01, 0.5, "Entire Dataset\n(100% N=50,000)", bbox=dict(boxstyle="square,pad=0.5", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.5), fontsize=9.5, fontweight="bold", va="center", ha="center")
    ax.annotate("", xy=(0.20, 0.5), xytext=(0.08, 0.5), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2))
    ax.text(0.33, 0.5, "Fit StandardScaler() / Imputer on ALL Data\n(Global μ & σ computed using validation/test instances)", bbox=dict(boxstyle="round,pad=0.5", facecolor="#FECACA", edgecolor="#EF4444", lw=1.5), fontsize=9.5, va="center", ha="center")
    ax.annotate("", xy=(0.60, 0.5), xytext=(0.52, 0.5), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2))
    ax.text(0.69, 0.5, "K-Fold Split or Holdout Split\n(Data already contaminated)", bbox=dict(boxstyle="square,pad=0.5", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.5), fontsize=9.5, va="center", ha="center")
    ax.annotate("", xy=(0.83, 0.5), xytext=(0.80, 0.5), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2))
    ax.text(0.92, 0.5, "Optimistic Biased Score\n(Crashes in Production!)", bbox=dict(boxstyle="round,pad=0.5", facecolor="#DC2626", edgecolor="#991B1B", lw=2), fontsize=9.5, fontweight="bold", color="white", va="center", ha="center")

    # Subplot B: The Vault-Lock Holdout Strategy (Image 2)
    ax = axes[1]
    ax.axis("off")
    ax.set_title("B. Vault-Lock & Encapsulated Pipeline Strategy (Image 2 Implementation)", fontsize=12, fontweight="bold", loc="left", color="#0D9488")
    ax.text(0.01, 0.5, "Raw Placement Data\n(N=50,000 / Sample)", bbox=dict(boxstyle="square,pad=0.5", facecolor="#CCFBF1", edgecolor="#0D9488", lw=1.5), fontsize=9.5, fontweight="bold", va="center", ha="center")
    ax.annotate("", xy=(0.14, 0.72), xytext=(0.08, 0.55), arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.8))
    ax.annotate("", xy=(0.14, 0.28), xytext=(0.08, 0.45), arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.8))
    ax.text(0.26, 0.72, "Development Set (80%):\nStratified 4-Fold CV", bbox=dict(boxstyle="round,pad=0.4", facecolor="#E0F2FE", edgecolor="#0284C7", lw=1.5), fontsize=9, va="center", ha="center")
    ax.text(0.26, 0.28, "Vault Test Set (20%):\nLOCKED IN VAULT (Untouched)", bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#D97706", lw=1.5), fontsize=9, fontweight="bold", va="center", ha="center")
    ax.annotate("", xy=(0.42, 0.72), xytext=(0.38, 0.72), arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.5))
    ax.text(0.56, 0.72, "make_pipeline(StandardScaler(), Model)\nFit ONLY on Train Folds; Transform Val Folds", bbox=dict(boxstyle="square,pad=0.5", facecolor="#E0F2FE", edgecolor="#0284C7", lw=1.5), fontsize=9, va="center", ha="center")
    ax.annotate("", xy=(0.82, 0.5), xytext=(0.74, 0.72), arrowprops=dict(arrowstyle="->", color="#0D9488", lw=2))
    ax.annotate("", xy=(0.82, 0.5), xytext=(0.40, 0.28), arrowprops=dict(arrowstyle="->", color="#D97706", lw=2))
    ax.text(0.92, 0.5, "Single Final Evaluation\non Vault Test Set\n(True Generalization Acc)", bbox=dict(boxstyle="round,pad=0.5", facecolor="#059669", edgecolor="#065F46", lw=2), fontsize=9.5, fontweight="bold", color="white", va="center", ha="center")

    # Subplot C: Nested Cross-Validation (Image 1)
    ax = axes[2]
    ax.axis("off")
    ax.set_title("C. Nested Cross-Validation for Tuning without Snooping (Image 1 Implementation)", fontsize=12, fontweight="bold", loc="left", color="#4338CA")
    ax.text(0.01, 0.5, "Outer 5-Fold Stratified CV\n(Unbiased Performance)", bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF2FF", edgecolor="#4F46E5", lw=1.5), fontsize=9.5, fontweight="bold", va="center", ha="center")
    ax.annotate("", xy=(0.17, 0.5), xytext=(0.11, 0.5), arrowprops=dict(arrowstyle="->", color="#4F46E5", lw=2))
    ax.text(0.28, 0.5, "Outer Train Fold (K-1)\n(Isolated for Tuning)", bbox=dict(boxstyle="square,pad=0.4", facecolor="#E0E7FF", edgecolor="#6366F1", lw=1.5), fontsize=9, va="center", ha="center")
    ax.annotate("", xy=(0.43, 0.5), xytext=(0.39, 0.5), arrowprops=dict(arrowstyle="->", color="#6366F1", lw=2))
    ax.text(0.57, 0.5, "Inner 3-Fold Stratified CV (GridSearchCV)\nFinds Best C in [0.01, 0.1, 1.0, 10.0]\nUsing Train Fold Only", bbox=dict(boxstyle="round,pad=0.5", facecolor="#C7D2FE", edgecolor="#4338CA", lw=1.5), fontsize=9, fontweight="bold", va="center", ha="center")
    ax.annotate("", xy=(0.77, 0.5), xytext=(0.74, 0.5), arrowprops=dict(arrowstyle="->", color="#4338CA", lw=2))
    ax.text(0.90, 0.5, "Evaluate Best Model on\nUnseen Outer Test Fold\n(Unbiased Nested F1)", bbox=dict(boxstyle="round,pad=0.5", facecolor="#4338CA", edgecolor="#312E81", lw=2), fontsize=9.5, fontweight="bold", color="white", va="center", ha="center")

    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35)
    save_figure(fig, "01_leakage_concept_architecture.png", dirs)
    print("Saved: 01_leakage_concept_architecture.png")

    # --------------------------------------------------------------------------
    # PLOT 2: LEAKY VS LEAKAGE-FREE PERFORMANCE BENCHMARK
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    leaky_means = [comparison_results[m]['leaky_mean'] for m in metrics]
    clean_means = [comparison_results[m]['clean_mean'] for m in metrics]
    gaps = [comparison_results[m]['gap'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    # Bar chart comparison
    axes[0].bar(x - width/2, leaky_means, width, label='Leaky (Global Preprocessing)', color='#EF4444', alpha=0.9, edgecolor='black', linewidth=0.5)
    axes[0].bar(x + width/2, clean_means, width, label='Leakage-Free (Pipeline)', color='#0D9488', alpha=0.9, edgecolor='black', linewidth=0.5)

    axes[0].set_ylabel('Cross-Validation Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Placement Model: Leaky vs. Leakage-Free Pipeline', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
    axes[0].set_ylim([0.80, 1.00])
    axes[0].legend(loc='lower right', frameon=True, fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for i, (l_val, c_val) in enumerate(zip(leaky_means, clean_means)):
        axes[0].text(x[i] - width/2, l_val + 0.005, f"{l_val:.3f}", ha='center', fontsize=8.5, fontweight='bold')
        axes[0].text(x[i] + width/2, c_val + 0.005, f"{c_val:.3f}", ha='center', fontsize=8.5, fontweight='bold')

    # Hastie-Tibshirani Feature Selection Leakage Proof (The extreme leakage case)
    # Demonstrate how selecting features before CV creates false accuracy
    np.random.seed(42)
    n_samples, n_noise_feats = 200, 500
    X_noise = np.random.randn(n_samples, n_noise_feats)
    y_noise = np.random.randint(0, 2, n_samples)
    
    # Leaky selection: correlate on ALL samples first, pick top 10
    corrs = np.array([abs(np.corrcoef(X_noise[:, col], y_noise)[0, 1]) for col in range(n_noise_feats)])
    top_indices = np.argsort(corrs)[-10:]
    X_leaky_selected = X_noise[:, top_indices]
    
    cv_noisy_leaky = cross_val_score(LogisticRegression(), X_leaky_selected, y_noise, cv=5).mean()
    
    # Leakage-free: inside pipeline
    from sklearn.feature_selection import SelectKBest, f_classif
    pipe_noisy = Pipeline([
        ('select', SelectKBest(f_classif, k=10)),
        ('model', LogisticRegression())
    ])
    cv_noisy_clean = cross_val_score(pipe_noisy, X_noise, y_noise, cv=5).mean()

    demo_x = [0, 1]
    demo_scores = [cv_noisy_leaky, cv_noisy_clean]
    demo_labels = ['Leaky Selection\n(Select BEFORE CV)', 'Leak-Free Pipeline\n(Select INSIDE CV)']
    demo_colors = ['#DC2626', '#059669']

    axes[1].bar(demo_x, demo_scores, color=demo_colors, width=0.45, edgecolor='black', linewidth=0.5)
    axes[1].axhline(0.50, color='gray', linestyle='--', linewidth=1.5, label='Random Guess Baseline (50%)')
    axes[1].set_ylabel('Accuracy on Pure Random Noise', fontsize=11, fontweight='bold')
    axes[1].set_title('Textbook Data Leakage Proof (Hastie & Tibshirani)\nFeature Selection on Pure Noise (Truth = 50%)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(demo_x)
    axes[1].set_xticklabels(demo_labels, fontsize=10, fontweight='bold')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(loc='upper right', fontsize=9.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    axes[1].text(0, cv_noisy_leaky + 0.02, f"{cv_noisy_leaky*100:.1f}%\n(FALSE ILLUSION)", ha='center', fontsize=9.5, fontweight='bold', color='#DC2626')
    axes[1].text(1, cv_noisy_clean + 0.02, f"{cv_noisy_clean*100:.1f}%\n(HONEST CHANCE)", ha='center', fontsize=9.5, fontweight='bold', color='#059669')

    plt.tight_layout()
    save_figure(fig, "02_leaky_vs_leakage_free_benchmark.png", dirs)
    print("Saved: 02_leaky_vs_leakage_free_benchmark.png")

    # --------------------------------------------------------------------------
    # PLOT 3: MATHEMATICAL DISTRIBUTION CONTAMINATION PROOF
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Empirical Proof of Data Contamination: Global Scaler vs Train-Only Scaler", fontsize=15, fontweight="bold")

    features_to_plot = ['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating']
    for idx, feat in enumerate(features_to_plot):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # Find contamination stats
        c_stat = next(item for item in contamination_data if item["feature"] == feat)
        
        labels = ['Global Fit (μ_all)', 'Fold Train Fit (μ_tr)']
        values = [c_stat['mu_global'], c_stat['mu_train']]
        stds = [c_stat['sig_global'], c_stat['sig_train']]

        x_pos = np.arange(2)
        ax.bar(x_pos, values, yerr=stds, capsize=6, color=['#EF4444', '#0D9488'], width=0.4, alpha=0.85, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_title(f"{feat}\nΔμ={c_stat['mu_shift']:.4f} | Max z-shift={c_stat['max_z_distortion']:.4f}", fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature Mean ± σ', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_figure(fig, "03_distribution_contamination_proof.png", dirs)
    print("Saved: 03_distribution_contamination_proof.png")

    # --------------------------------------------------------------------------
    # PLOT 4: NESTED CROSS-VALIDATION (IMAGE 1 PERFORMANCE & FOLD DRIFT)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    outer_folds = np.arange(1, len(nested_results["outer_f1_scores"]) + 1)
    f1_scores = nested_results["outer_f1_scores"]
    acc_scores = nested_results["outer_acc_scores"]
    f1_mean = nested_results["nested_f1_mean"]
    f1_std = nested_results["nested_f1_std"]

    # Outer fold metrics
    axes[0].plot(outer_folds, f1_scores, marker='o', linewidth=2.5, markersize=8, color='#4F46E5', label='Outer Fold F1-Score')
    axes[0].plot(outer_folds, acc_scores, marker='s', linewidth=2.5, markersize=8, color='#059669', label='Outer Fold Accuracy')
    axes[0].axhline(f1_mean, color='#4F46E5', linestyle='--', linewidth=1.5, label=f'Mean F1: {f1_mean:.4f}')
    axes[0].fill_between(outer_folds, f1_mean - 2*f1_std, f1_mean + 2*f1_std, color='#4F46E5', alpha=0.15, label='95% Unbiased Interval (±2σ)')
    axes[0].set_xlabel('Outer Cross-Validation Fold', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Nested CV: Outer Fold Unbiased Generalization', fontsize=13, fontweight='bold')
    axes[0].set_xticks(outer_folds)
    axes[0].legend(loc='lower left', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Inner vs Outer Fold Comparison (Tuning Snooping Proof)
    inner_best_scores = [d["inner_cv_best_f1"] for d in nested_results["fold_details"]]
    outer_scores = [d["outer_val_f1"] for d in nested_results["fold_details"]]
    best_Cs = [str(d["best_C"]) for d in nested_results["fold_details"]]

    x_folds = np.arange(len(outer_folds))
    w = 0.35
    axes[1].bar(x_folds - w/2, inner_best_scores, w, label='Inner CV Best F1 (Tuned on Train)', color='#F59E0B', alpha=0.9, edgecolor='black', linewidth=0.5)
    axes[1].bar(x_folds + w/2, outer_scores, w, label='Outer Unseen Test Fold F1', color='#4F46E5', alpha=0.9, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x_folds)
    axes[1].set_xticklabels([f"Fold {i+1}\n(C={best_Cs[i]})" for i in range(len(outer_folds))], fontsize=10, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Inner CV Tuned Score vs. Unseen Outer Validation', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "04_nested_cv_outer_vs_inner_performance.png", dirs)
    print("Saved: 04_nested_cv_outer_vs_inner_performance.png")

    # --------------------------------------------------------------------------
    # PLOT 5: VAULT-LOCK STRATEGY EVALUATION (IMAGE 2 SUITE)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Dev CV Folds vs Unseen Vault Test Score
    dev_folds = np.arange(1, len(vault_results["dev_cv_scores"]) + 1)
    dev_scores = vault_results["dev_cv_scores"]
    axes[0].bar(dev_folds, dev_scores, color='#0284C7', alpha=0.85, width=0.45, edgecolor='black', label='Dev Set Fold Accuracy')
    axes[0].axhline(vault_results["dev_cv_mean"], color='#0284C7', linestyle='--', linewidth=2, label=f'Dev Mean CV: {vault_results["dev_cv_mean"]:.4f}')
    axes[0].axhline(vault_results["test_accuracy"], color='#10B981', linestyle='-', linewidth=2.5, label=f'Vault Test Accuracy: {vault_results["test_accuracy"]:.4f}')
    axes[0].set_xlabel('Dev Set 4-Fold CV', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Dev Cross-Validation vs. Vault Test Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xticks(dev_folds)
    axes[0].set_ylim([min(dev_scores) - 0.05, max(dev_scores) + 0.05])
    axes[0].legend(loc='lower right', fontsize=9.5)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Panel 2: Confusion Matrix on Unseen Vault Test Set
    cm = np.array(vault_results["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                xticklabels=['Not Placed (0)', 'Placed (1)'],
                yticklabels=['Not Placed (0)', 'Placed (1)'],
                annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Confusion Matrix on Vault Test Set (N={cm.sum():,})', fontsize=12, fontweight='bold')

    # Panel 3: ROC Curve on Unseen Vault Test Set
    axes[2].plot(vault_results["fpr"], vault_results["tpr"], color='#0D9488', linewidth=2.5,
                 label=f'Vault Test ROC (AUC = {vault_results["test_auc"]:.4f})')
    axes[2].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Chance (AUC = 0.50)')
    axes[2].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axes[2].set_title('ROC Curve on Unseen Vault Test Set', fontsize=12, fontweight='bold')
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "05_vault_lock_test_evaluation.png", dirs)
    print("Saved: 05_vault_lock_test_evaluation.png")


# ==============================================================================
# REPORT & SUMMARY EXPORT
# ==============================================================================
def export_reports(vault_results, nested_results, comparison_results, contamination_data, dirs):
    """Write comprehensive textual and JSON analytical reports."""
    # JSON payload for web and API
    summary_data = {
        "title": "Data Leakage Prevention & Robust Model Evaluation (Session 28)",
        "features": ['CGPA', 'AptitudeTestScore', 'CodingTestScore', 'SoftSkillsRating', 'Internships', 'Projects'],
        "vault_lock": vault_results,
        "nested_cv": nested_results,
        "leakage_benchmark": comparison_results,
        "contamination": contamination_data
    }

    json_path = os.path.join(dirs["base"], "leakage_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    # Also copy to outputs dir
    with open(os.path.join(dirs["outputs"], "leakage_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # Text report
    report_path = os.path.join(dirs["base"], "leakage_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA LEAKAGE PREVENTION & ROBUST MODEL EVALUATION REPORT (SESSION 28)\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. WORKFLOW 1: VAULT-LOCK & STRATIFIED DEV PIPELINE (IMAGE 2)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Development CV Accuracy: {vault_results['dev_cv_mean']:.4f} (+/- {vault_results['dev_cv_std']*2:.4f})\n")
        f.write(f"Mean Development CV F1-Score: {vault_results['dev_cv_f1_mean']:.4f}\n")
        f.write(f"Mean Development CV ROC-AUC:  {vault_results['dev_cv_auc_mean']:.4f}\n")
        f.write(f"Unseen Vault Test Accuracy:   {vault_results['test_accuracy']:.4f}\n")
        f.write(f"Unseen Vault Test F1-Score:   {vault_results['test_f1']:.4f}\n")
        f.write(f"Unseen Vault Test ROC-AUC:    {vault_results['test_auc']:.4f}\n\n")

        f.write("2. WORKFLOW 2: NESTED CROSS-VALIDATION & HYPERPARAMETER TUNING (IMAGE 1)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Outer 5-Fold F1 Scores:       {np.array(nested_results['outer_f1_scores']).round(4)}\n")
        f.write(f"Unbiased Nested CV F1-Score:  {nested_results['nested_f1_mean']:.4f} (+/- {nested_results['nested_f1_std']*2:.4f})\n")
        f.write(f"Outer 5-Fold Accuracies:      {np.array(nested_results['outer_acc_scores']).round(4)}\n")
        f.write(f"Unbiased Nested CV Accuracy:  {nested_results['nested_acc_mean']:.4f} (+/- {nested_results['nested_acc_std']*2:.4f})\n")
        f.write(f"Non-Nested Naive Best CV F1:  {nested_results['non_nested_optimistic_f1']:.4f}\n")
        f.write(f"Tuning Optimism Bias (Gap):   {nested_results['tuning_optimism_delta']:+.4f}\n\n")

        f.write("3. WORKFLOW 3: LEAKY VS. LEAKAGE-FREE COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<15} | {'Leaky':<12} | {'Leakage-Free':<14} | {'Optimism Bias (Gap)':<20}\n")
        for m, data in comparison_results.items():
            f.write(f"{m.capitalize():<15} | {data['leaky_mean']:.4f}       | {data['clean_mean']:.4f}         | {data['gap']:+.4f}\n")

        f.write("\n4. WORKFLOW 4: MATHEMATICAL CONTAMINATION ANALYSIS (FEATURE DRIFT)\n")
        f.write("-" * 80 + "\n")
        for c in contamination_data:
            f.write(f"Feature: {c['feature']:<20} | Mean Shift: {c['mu_shift']:.4f} | Std Shift: {c['sig_shift']:.4f} | Max z-distortion: {c['max_z_distortion']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    # Mirror report to outputs dir
    with open(os.path.join(dirs["outputs"], "leakage_report.txt"), "w") as f:
        with open(report_path, "r") as src_f:
            f.write(src_f.read())

    print("Reports successfully generated and mirrored.")


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def run_data_leakage_analysis(sample_size=5000, random_state=42):
    """
    Main orchestrator for Session 28 Data Leakage Analysis.
    Executes both workflows from user specification, runs benchmarks,
    generates publication plots, and exports telemetry reports.
    """
    print("=" * 80)
    print("SESSION 28: DATA LEAKAGE PREVENTION, VAULT-LOCKING & NESTED CV")
    print("=" * 80)

    dirs = setup_directories()
    print(f"Artifact directories initialized at: {dirs['base']}")

    # 1. Load data
    X, y, df = load_dataset(sample_size=sample_size, random_state=random_state)
    print(f"Dataset loaded: {len(X):,} students | Target: 1={y.mean()*100:.1f}%, 0={(1-y.mean())*100:.1f}%")

    # 2. Workflow 1: Vault-Lock & Dev CV (Image 2)
    vault_results, pipeline, splits = run_vault_lock_workflow(X, y, dirs, random_state=random_state)

    # 3. Workflow 2: Nested Cross-Validation (Image 1)
    nested_results = run_nested_cv_workflow(X, y, dirs, random_state=random_state)

    # 4. Workflow 3: Leaky vs Leakage-Free Benchmark
    comparison_results = run_leakage_comparison_experiment(X, y, random_state=random_state)

    # 5. Workflow 4: Mathematical Contamination Analysis
    contamination_data, _ = analyze_distribution_contamination(X, y, random_state=random_state)

    # 6. Generate All 5 High-Resolution Charts
    print("\n>>> GENERATING PUBLICATION PLOTS...")
    generate_all_plots(vault_results, nested_results, comparison_results, contamination_data, dirs)

    # 7. Export Reports & JSON Summary
    export_reports(vault_results, nested_results, comparison_results, contamination_data, dirs)

    print("\n" + "=" * 80)
    print("SESSION 28: DATA LEAKAGE ANALYSIS COMPLETE SUCCESSFULLY!")
    print(f"Generated 5 plots in {dirs['plots']}")
    print(f"Generated report at {os.path.join(dirs['base'], 'leakage_report.txt')}")
    print("=" * 80)

    return {
        "vault_results": vault_results,
        "nested_results": nested_results,
        "comparison_results": comparison_results,
        "contamination_data": contamination_data
    }


if __name__ == "__main__":
    run_data_leakage_analysis()
