# Principal Component Analysis (PCA) - Session 26
# ==============================================================================
# Sessions 21 and 24 both clustered on the full set of scaled numeric
# features and only used PCA as an afterthought, to compress results
# down to 2D for plotting. This session makes PCA itself the subject:
# what it actually does, how much of the data's structure a handful
# of components can capture, and why "compress to 2D for plotting"
# was a reasonable thing to do at all.
#
# What PCA does: find a NEW set of axes (principal components) that
#   are linear combinations of the original features, chosen so that
#   the first axis captures as much of the data's variance as
#   possible, the second axis captures as much of the REMAINING
#   variance as possible while being perpendicular (orthogonal) to
#   the first, and so on. The components are ordered by how much
#   variance each one explains - the data hasn't changed, only the
#   coordinate system used to describe it has.
#
# Why variance is the thing being maximized: directions where the
#   data barely spreads out contain little information (every student
#   looks similar along that direction); directions where it spreads
#   out a lot are where the real differences between students live.
#   PCA assumes "spread = information" and rotates the axes to point
#   straight at where the information actually is.
#
# Dimensionality reduction: keeping only the first few components
#   (instead of all of them) throws away the directions with the
#   least variance, on the assumption they were mostly noise. This is
#   lossy - some genuine information is discarded too - but often a
#   small number of components explain most of the variance, meaning
#   little is actually lost.

import sys
import os
import json
import shutil

# Robust path handling across execution environments
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(models_dir)
app_dir = os.path.join(src_dir, "05_Web_Application")

for path_dir in [app_dir, src_dir, models_dir, current_dir]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config


def run_pca_analysis():
    print("=" * 70, flush=True)
    print("Starting Principal Component Analysis (Session 26)...", flush=True)
    print("=" * 70, flush=True)

    # Ensure destination output directories exist
    pca_dir = getattr(config, "PCA_DIR", current_dir)
    pca_outputs_dir = getattr(config, "PCA_OUTPUTS_DIR", os.path.join(src_dir, "06_Outputs_and_Utils", "outputs", "pca"))
    web_plots_dir = os.path.join(app_dir, "static", "plots", "pca")
    plots_dir = os.path.join(pca_dir, "plots")

    for target_dir in [pca_dir, plots_dir, pca_outputs_dir, web_plots_dir]:
        os.makedirs(target_dir, exist_ok=True)

    # ---------- Load, impute, scale - identical pipeline to Sessions 21 & 24 ----------
    train_path = os.path.join(getattr(config, "SPLITS_DIR", ""), "train.csv")
    if not os.path.exists(train_path):
        alt_train_path = os.path.join(src_dir, "06_Outputs_and_Utils", "splits", "train.csv")
        if os.path.exists(alt_train_path):
            train_path = alt_train_path
        elif hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
            from sklearn.model_selection import train_test_split
            raw_df = pd.read_csv(config.RAW_DATA_PATH)
            train_df, _ = train_test_split(raw_df, test_size=0.3, random_state=config.RANDOM_STATE)
            train_df.to_csv(train_path, index=False)

    df = pd.read_csv(train_path)
    feature_cols = config.NUMERIC_COLUMNS
    x_raw = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(imputer.fit_transform(x_raw), columns=feature_cols, index=x_raw.index)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    print(f"Loaded {len(df):,} students, {len(feature_cols)} scaled numeric features", flush=True)

    # ---------- Fit PCA keeping ALL components first ----------
    pca_full = PCA(n_components=None, random_state=config.RANDOM_STATE)
    pca_full.fit(x_scaled)

    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print(f"\nTotal components (= number of input features): {len(explained_variance_ratio)}", flush=True)
    print("\nVariance explained by first 10 components:", flush=True)
    for i in range(min(10, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {round(explained_variance_ratio[i]*100, 2)}%  (cumulative: {round(cumulative_variance[i]*100, 2)}%)", flush=True)

    # How many components needed to reach common variance thresholds
    thresholds_dict = {}
    for threshold in [0.80, 0.90, 0.95]:
        n_needed = int(np.searchsorted(cumulative_variance, threshold) + 1)
        thresholds_dict[f"{int(threshold*100)}%"] = n_needed
        print(f"Components needed for {int(threshold*100)}% cumulative variance: {n_needed}", flush=True)

    # Helper function to save figure across multiple configured target locations
    def save_chart(filename):
        targets = [
            os.path.join(pca_dir, filename),
            os.path.join(plots_dir, filename),
            os.path.join(pca_outputs_dir, filename),
            os.path.join(web_plots_dir, filename)
        ]
        for target in targets:
            plt.savefig(target, dpi=300, bbox_inches="tight")

    # ---------- Scree plot: individual and cumulative variance ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_show = min(20, len(explained_variance_ratio))
    axes[0].bar(range(1, n_show + 1), explained_variance_ratio[:n_show] * 100, color="#4C72B0")
    axes[0].set_xlabel("Principal Component", fontsize=11)
    axes[0].set_ylabel("Variance Explained (%)", fontsize=11)
    axes[0].set_title("Scree Plot: Variance per Component", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(1, n_show + 1, 2))
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].plot(range(1, n_show + 1), cumulative_variance[:n_show] * 100, marker="o", color="#DD8452", linewidth=2)
    for threshold in [80, 90, 95]:
        axes[1].axhline(y=threshold, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        axes[1].text(n_show - 1.5, threshold + 1, f"{threshold}%", color="#4b5563", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Number of Components", fontsize=11)
    axes[1].set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
    axes[1].set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(1, n_show + 1, 2))
    axes[1].grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_chart("scree_and_cumulative_variance.png")
    plt.close()
    print("Scree plot saved", flush=True)

    # ---------- Loadings interpretation: PC1 and PC2 ----------
    loadings = pd.DataFrame(
        pca_full.components_[:2].T,
        columns=["PC1", "PC2"],
        index=feature_cols,
    )
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(8)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(8)

    print("\nTop features contributing to PC1 (by absolute loading):", flush=True)
    for feature in top_pc1.index:
        print(f"  {feature:<20} {round(loadings.loc[feature, 'PC1'], 3)}", flush=True)
    print("\nTop features contributing to PC2 (by absolute loading):", flush=True)
    for feature in top_pc2.index:
        print(f"  {feature:<20} {round(loadings.loc[feature, 'PC2'], 3)}", flush=True)

    plt.figure(figsize=(8, 8))
    for feature in feature_cols:
        plt.arrow(0, 0, loadings.loc[feature, "PC1"], loadings.loc[feature, "PC2"],
                  head_width=0.02, color="#6E2436", alpha=0.6)
        plt.text(loadings.loc[feature, "PC1"] * 1.1, loadings.loc[feature, "PC2"] * 1.1,
                  feature, fontsize=7, ha="center")
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="gray")
    plt.gca().add_artist(circle)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(f"PC1 ({round(explained_variance_ratio[0]*100, 1)}% variance)", fontsize=11)
    plt.ylabel(f"PC2 ({round(explained_variance_ratio[1]*100, 1)}% variance)", fontsize=11)
    plt.title("PCA Loadings: How Each Original Feature Maps onto PC1/PC2", fontsize=12, fontweight="bold")
    plt.gca().set_aspect("equal")
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_chart("loadings_biplot.png")
    plt.close()
    print("Loadings biplot saved", flush=True)

    # ---------- 2D projection, colored by placement ----------
    pc1_pc2_variance = cumulative_variance[1]
    print(f"\nPC1 + PC2 together capture {round(pc1_pc2_variance*100, 1)}% of total variance", flush=True)

    x_2d = pca_full.transform(x_scaled)[:, :2]
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=df[config.TARGET_COLUMN], cmap="coolwarm", s=6, alpha=0.45)
    plt.xlabel(f"PC1 ({round(explained_variance_ratio[0]*100, 1)}% variance)", fontsize=11)
    plt.ylabel(f"PC2 ({round(explained_variance_ratio[1]*100, 1)}% variance)", fontsize=11)
    plt.title(f"All {len(df):,} Students Projected onto PC1/PC2\n(colored by actual placement outcome, {round(pc1_pc2_variance*100, 1)}% of total variance captured)", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Placement Outcome (0 = Not Placed, 1 = Placed)", fontsize=10)
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_chart("pca_2d_by_placement.png")
    plt.close()
    print("2D projection colored by placement saved", flush=True)

    # ---------- Save Text Report ----------
    report_content = []
    report_content.append(f"PCA on {len(df):,} students, {len(feature_cols)} scaled, median-imputed\n")
    report_content.append("numeric features (same pipeline as Sessions 21 & 24).\n\n")
    report_content.append("Variance explained by first 10 components:\n")
    for i in range(min(10, len(explained_variance_ratio))):
        report_content.append(f"  PC{i+1}: {round(explained_variance_ratio[i]*100, 2)}%  (cumulative: {round(cumulative_variance[i]*100, 2)}%)\n")
    report_content.append("\nComponents needed for common variance thresholds:\n")
    for threshold in [0.80, 0.90, 0.95]:
        n_needed = int(np.searchsorted(cumulative_variance, threshold) + 1)
        report_content.append(f"  {int(threshold*100)}% cumulative variance: {n_needed} components\n")
    report_content.append(f"\nPC1 + PC2 together: {round(pc1_pc2_variance*100, 1)}% of total variance\n")
    report_content.append("(this is the number that quietly backed every 2D PCA cluster plot\n")
    report_content.append(" in Sessions 21 and 24 - now made explicit)\n\n")
    report_content.append("Top features contributing to PC1 (by absolute loading):\n")
    for feature in top_pc1.index:
        report_content.append(f"  {feature}: {round(loadings.loc[feature, 'PC1'], 3)}\n")
    report_content.append("\nTop features contributing to PC2 (by absolute loading):\n")
    for feature in top_pc2.index:
        report_content.append(f"  {feature}: {round(loadings.loc[feature, 'PC2'], 3)}\n")

    full_report_text = "".join(report_content)

    report_paths = [
        os.path.join(pca_dir, "pca_report.txt"),
        os.path.join(pca_outputs_dir, "pca_report.txt")
    ]
    for rpath in report_paths:
        with open(rpath, "w") as f:
            f.write(full_report_text)

    # ---------- Save Structured JSON Summary ----------
    summary_data = {
        "status": "success",
        "method": "Principal Component Analysis (PCA)",
        "total_students": len(df),
        "total_features": len(feature_cols),
        "total_components": len(explained_variance_ratio),
        "pc1_variance_pct": round(float(explained_variance_ratio[0] * 100), 2),
        "pc2_variance_pct": round(float(explained_variance_ratio[1] * 100), 2),
        "pc1_pc2_cumulative_pct": round(float(pc1_pc2_variance * 100), 2),
        "variance_by_component": [
            {
                "component": f"PC{i+1}",
                "variance_pct": round(float(explained_variance_ratio[i] * 100), 2),
                "cumulative_pct": round(float(cumulative_variance[i] * 100), 2)
            }
            for i in range(len(explained_variance_ratio))
        ],
        "thresholds": thresholds_dict,
        "top_pc1_loadings": [
            {"feature": feat, "loading": round(float(loadings.loc[feat, "PC1"]), 3)}
            for feat in top_pc1.index
        ],
        "top_pc2_loadings": [
            {"feature": feat, "loading": round(float(loadings.loc[feat, "PC2"]), 3)}
            for feat in top_pc2.index
        ],
        "plots": [
            {
                "id": "scree_variance",
                "filename": "pca/scree_and_cumulative_variance.png",
                "title": "Scree Plot & Cumulative Variance Curve",
                "caption": "Scree bar chart and cumulative variance progression across components with 80%, 90%, and 95% thresholds."
            },
            {
                "id": "loadings_biplot",
                "filename": "pca/loadings_biplot.png",
                "title": "PCA Loadings Biplot (PC1 vs PC2)",
                "caption": "Vector representation of 20 original features projected onto principal components PC1 and PC2."
            },
            {
                "id": "pca_2d_scatter",
                "filename": "pca/pca_2d_by_placement.png",
                "title": "2D Student Landscape by Placement Status",
                "caption": "All 35,000 students projected into 2D principal coordinate space, colored by actual placement outcome."
            }
        ]
    }

    json_paths = [
        os.path.join(pca_dir, "pca_summary.json"),
        os.path.join(pca_outputs_dir, "pca_summary.json")
    ]
    for jpath in json_paths:
        with open(jpath, "w") as f:
            json.dump(summary_data, f, indent=4)

    print(f"\nReport saved to {os.path.join(pca_dir, 'pca_report.txt')}", flush=True)
    print(f"Structured summary saved to {os.path.join(pca_dir, 'pca_summary.json')}", flush=True)
    print(f"Outputs synchronized to {pca_outputs_dir} and {web_plots_dir}", flush=True)
    print("=" * 70, flush=True)
    print("PCA Analysis completed successfully!", flush=True)
    print("=" * 70, flush=True)

    return summary_data


if __name__ == "__main__":
    run_pca_analysis()
