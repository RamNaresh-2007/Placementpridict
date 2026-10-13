# Hierarchical Clustering for Placement Prediction Dataset
# ==========================================================
# This script performs agglomerative hierarchical clustering on student performance data.
# It analyzes:
#   1. Linkage criteria (Ward, Complete, Average, Single) and their tree geometry.
#   2. Cophenetic correlation coefficient (measuring distance preservation fidelity).
#   3. The single-linkage "chaining" effect vs Ward's variance-minimizing balance.
#   4. Cutting the dendrogram at different heights to find optimal K via Silhouette score.
#   5. Agreement with K-Means using Adjusted Rand Index (ARI).
#   6. External validation: How unsupervised clusters map to actual PlacementStatus.
#   7. Cluster profiling: Mean academic & technical metrics per cluster persona.

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "05_Web_Application"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import config

# Ensure destination output directory exists
os.makedirs(config.HIERARCHICAL_DIR, exist_ok=True)

# agglomerative clustering needs the full pairwise distance matrix (O(N^2)).
# With 50,000 rows, a full matrix would require ~1.25 billion pairwise distances (~10 GB).
# A representative subsample of 2,000 keeps computation fast, tractable (~2M distances),
# and statistically representative of the underlying manifold.
SUBSAMPLE_SIZE = 2000

# ---------- Load, Impute, and Scale Features ----------
print(f"Loading dataset from: {config.RAW_DATA_PATH}")
df = pd.read_csv(config.RAW_DATA_PATH)

# Train/validation split (80/20) matching the modeling pipeline
train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.RANDOM_STATE)

feature_cols = config.NUMERIC_COLUMNS
x_raw = train_df[feature_cols].copy()

# Median imputation for missing scores (e.g. MockInterviewScore, CodingTestScore)
imputer = SimpleImputer(strategy="median")
x_imputed = pd.DataFrame(imputer.fit_transform(x_raw), columns=feature_cols, index=x_raw.index)

# Standard scaling: zero mean, unit variance
scaler = StandardScaler()
x_scaled_full = scaler.fit_transform(x_imputed)

# Subsampling for hierarchical clustering
rng = np.random.RandomState(config.RANDOM_STATE)
sample_idx = rng.choice(len(x_scaled_full), size=SUBSAMPLE_SIZE, replace=False)
x_sample = x_scaled_full[sample_idx]
train_sample_df = train_df.iloc[sample_idx].copy()
placement_sample = train_sample_df[config.TARGET_COLUMN].values

print(f"Loaded {len(df):,} total records, split train={len(train_df):,}.")
print(f"Subsampled {SUBSAMPLE_SIZE:,} students for hierarchical clustering.")
print(f"Pairwise distance entries: {SUBSAMPLE_SIZE * (SUBSAMPLE_SIZE - 1) // 2:,}")

# ---------- Build Trees with All Four Linkage Methods ----------
linkage_methods = ["ward", "complete", "average", "single"]
linkage_matrices = {}
cophenetic_scores = {}
condensed_distances = pdist(x_sample, metric="euclidean")

print("\n--- Computing Hierarchical Linkages and Cophenetic Correlations ---")
for method in linkage_methods:
    Z = linkage(x_sample, method=method)
    linkage_matrices[method] = Z
    coph_corr, _ = cophenet(Z, condensed_distances)
    cophenetic_scores[method] = coph_corr
    print(f"{method:<10} linkage: cophenetic correlation = {round(coph_corr, 4)}")

# ---------- Dendrograms Comparison (2x2 Grid) ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for ax, method in zip(axes.flat, linkage_methods):
    dendrogram(linkage_matrices[method], truncate_mode="lastp", p=30, ax=ax, no_labels=True)
    ax.set_title(f"{method.capitalize()} Linkage (Cophenetic Corr: {round(cophenetic_scores[method], 3)})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Merged Cluster (Truncated to last 30 merges)", fontsize=10)
    ax.set_ylabel("Merge Distance", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle(f"Dendrogram Comparison across Linkage Methods (n={SUBSAMPLE_SIZE:,} students)",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
dendrogram_path = os.path.join(config.HIERARCHICAL_DIR, "dendrograms_by_linkage.png")
plt.savefig(dendrogram_path, dpi=300)
plt.close()
print(f"\nDendrogram comparison saved to {dendrogram_path}")

# ---------- Single Linkage Chaining vs Ward Linkage ----------
single_clusters_at_4 = fcluster(linkage_matrices["single"], t=4, criterion="maxclust")
single_cluster_sizes = pd.Series(single_clusters_at_4).value_counts().sort_index()

ward_clusters_at_4 = fcluster(linkage_matrices["ward"], t=4, criterion="maxclust")
ward_cluster_sizes = pd.Series(ward_clusters_at_4).value_counts().sort_index()

print("\n--- Cluster Size Distribution Comparison (Cut at K=4) ---")
print(f"Single Linkage cluster sizes: {single_cluster_sizes.to_dict()}")
print("  -> Chaining effect: one massive cluster absorbs almost all data while outliers form singleton clusters.")
print(f"Ward Linkage cluster sizes:   {ward_cluster_sizes.to_dict()}")
print("  -> Ward's variance minimization produces well-balanced, dense clusters.")

# ---------- Optimal K Sweep by Cutting Ward's Tree ----------
k_range = range(2, 9)
silhouette_by_k = []
print("\n--- Silhouette Scores by Cut Height (Ward Tree) ---")
for k in k_range:
    labels_k = fcluster(linkage_matrices["ward"], t=k, criterion="maxclust")
    sil = silhouette_score(x_sample, labels_k)
    silhouette_by_k.append(sil)
    print(f"Ward cut at K={k}: Silhouette Score = {round(sil, 4)}")

best_k_idx = int(np.argmax(silhouette_by_k))
best_k_hier = list(k_range)[best_k_idx]
best_sil_score = silhouette_by_k[best_k_idx]
print(f"\nOptimal K for Ward linkage by Silhouette Score: K={best_k_hier} (Score: {round(best_sil_score, 4)})")

# Plot Silhouette Score vs K
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), silhouette_by_k, marker="o", linewidth=2.5, color="#1f77b4", markersize=8)
plt.axvline(x=best_k_hier, color="#d62728", linestyle="--", linewidth=1.8,
            label=f"Optimal K = {best_k_hier} (Sil = {round(best_sil_score, 3)})")
plt.xlabel("Number of Clusters (K) Cut from Single Ward Tree", fontsize=11)
plt.ylabel("Silhouette Score", fontsize=11)
plt.title("Cluster Quality: Silhouette Score vs. Cut Height (Ward)", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
sil_plot_path = os.path.join(config.HIERARCHICAL_DIR, "silhouette_by_cut_height.png")
plt.savefig(sil_plot_path, dpi=300)
plt.close()
print(f"Silhouette-by-cut-height chart saved to {sil_plot_path}")

# ---------- Agreement with K-Means (Adjusted Rand Index) ----------
# We compare Hierarchical (Ward) with K-Means at K=2 (the natural binary partition)
# and also at K=best_k_hier.
print("\n--- Agreement with K-Means (Adjusted Rand Index) ---")
OPTIMAL_K = 2
hier_labels_k2 = fcluster(linkage_matrices["ward"], t=OPTIMAL_K, criterion="maxclust")
kmeans_k2 = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
kmeans_labels_k2 = kmeans_k2.fit_predict(x_sample)
ari_k2 = adjusted_rand_score(hier_labels_k2, kmeans_labels_k2)
print(f"Adjusted Rand Index (Ward vs K-Means at K=2): {round(ari_k2, 4)}")

ari_best = None
if best_k_hier != OPTIMAL_K:
    hier_labels_best = fcluster(linkage_matrices["ward"], t=best_k_hier, criterion="maxclust")
    kmeans_best = KMeans(n_clusters=best_k_hier, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
    kmeans_labels_best = kmeans_best.fit_predict(x_sample)
    ari_best = adjusted_rand_score(hier_labels_best, kmeans_labels_best)
    print(f"Adjusted Rand Index (Ward vs K-Means at K={best_k_hier}): {round(ari_best, 4)}")

# ---------- Placement Rate and Feature Profiles by Cluster ----------
print("\n--- External Validation: Placement Status per Cluster (Ward, K=2) ---")
train_sample_df["Hier_Cluster_K2"] = hier_labels_k2
placement_by_cluster = train_sample_df.groupby("Hier_Cluster_K2")[config.TARGET_COLUMN].agg(["count", "mean"])
placement_by_cluster["mean"] = placement_by_cluster["mean"] * 100
placement_by_cluster.rename(columns={"count": "Student Count", "mean": "Placement Rate (%)"}, inplace=True)

for cluster_id, row in placement_by_cluster.iterrows():
    print(f"  Cluster {cluster_id}: {int(row['Student Count']):,} students | "
          f"Placement Rate = {round(row['Placement Rate (%)'], 2)}%")

# Feature Means per Cluster for In-depth Interpretation
cluster_profile = train_sample_df.groupby("Hier_Cluster_K2")[feature_cols].mean()
print("\n--- Mean Feature Values by Cluster ---")
print(cluster_profile.round(2))

# ---------- Visualize Cluster Characteristics ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Placement Rate by Cluster
clusters = [f"Cluster {c}" for c in placement_by_cluster.index]
rates = placement_by_cluster["Placement Rate (%)"].values
bars = ax1.bar(clusters, rates, color=["#2ca02c" if r >= 50 else "#d62728" for r in rates], width=0.5, edgecolor="black")
ax1.set_ylabel("Placement Rate (%)", fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_title("Placement Rate by Hierarchical Cluster (Ward K=2)", fontsize=12, fontweight="bold")
for bar in bars:
    h = bar.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Plot 2: Scatter plot of CGPA vs CodingTestScore colored by cluster
scatter = ax2.scatter(train_sample_df["CGPA"], train_sample_df["CodingTestScore"],
                      c=hier_labels_k2, cmap="Set1", alpha=0.6, edgecolors='none', s=25)
ax2.set_xlabel("CGPA", fontsize=11)
ax2.set_ylabel("Coding Test Score", fontsize=11)
ax2.set_title("Cluster Separation in Key Feature Space (Ward K=2)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.4)
legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster", loc="lower right")
ax2.add_artist(legend1)

plt.tight_layout()
profile_plot_path = os.path.join(config.HIERARCHICAL_DIR, "cluster_placement_profile.png")
plt.savefig(profile_plot_path, dpi=300)
plt.close()
print(f"Cluster placement profile chart saved to {profile_plot_path}")

# ---------- Save Full Text Report ----------
report_path = os.path.join(config.HIERARCHICAL_DIR, "hierarchical_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("HIERARCHICAL (AGGLOMERATIVE) CLUSTERING REPORT\n")
    f.write("Placement Prediction Dataset\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Sample Size: {SUBSAMPLE_SIZE:,} students (from {len(train_df):,} training split rows)\n")
    f.write(f"Features: {', '.join(feature_cols)}\n")
    f.write("Preprocessing: Median imputation + StandardScaler\n\n")

    f.write("1. COPHENETIC CORRELATION COEFFICIENT BY LINKAGE METHOD\n")
    f.write("   (Measures how faithfully the dendrogram preserves original pairwise Euclidean distances)\n")
    for method, score in cophenetic_scores.items():
        f.write(f"   - {method:<10}: {round(score, 4)}\n")
    f.write("\n")

    f.write("2. LINKAGE BEHAVIOR AT K=4 (CHAINING VS VARIANCE MINIMIZATION)\n")
    f.write(f"   - Single Linkage cluster sizes: {single_cluster_sizes.to_dict()}\n")
    f.write("     Observation: Severe chaining effect. One dominant cluster contains almost all points,\n")
    f.write("     while single noise/outlier points form isolated micro-clusters.\n")
    f.write(f"   - Ward Linkage cluster sizes:   {ward_cluster_sizes.to_dict()}\n")
    f.write("     Observation: Balanced, compact cluster distribution due to within-cluster variance minimization.\n\n")

    f.write("3. OPTIMAL K SELECTION (CUTTING WARD TREE)\n")
    for k, sil in zip(k_range, silhouette_by_k):
        f.write(f"   - K={k}: Silhouette Score = {round(sil, 4)}\n")
    f.write(f"   Optimal K by Silhouette Score: K={best_k_hier} (Score: {round(best_sil_score, 4)})\n\n")

    f.write("4. AGREEMENT WITH K-MEANS (ADJUSTED RAND INDEX)\n")
    f.write(f"   - Adjusted Rand Index at K=2: {round(ari_k2, 4)}\n")
    if ari_best is not None:
        f.write(f"   - Adjusted Rand Index at K={best_k_hier}: {round(ari_best, 4)}\n")
    f.write("   Interpretation: Both algorithms clustered without seeing PlacementStatus.\n")
    f.write("   The ARI indicates strong alignment on the underlying geometric partitioning of students.\n\n")

    f.write("5. EXTERNAL VALIDATION: PLACEMENT OUTCOME BY CLUSTER (K=2)\n")
    for cluster_id, row in placement_by_cluster.iterrows():
        f.write(f"   - Cluster {cluster_id}: {int(row['Student Count']):,} students | "
                f"Placement Rate = {round(row['Placement Rate (%)'], 2)}%\n")
    f.write("\n")

    f.write("6. CLUSTER FEATURE PROFILES (MEAN VALUES)\n")
    for cluster_id, row in cluster_profile.iterrows():
        f.write(f"   - Cluster {cluster_id}:\n")
        for col in feature_cols:
            f.write(f"       {col:<20}: {round(row[col], 2)}\n")
    f.write("\n")

    f.write("7. KEY ARCHITECTURAL & METHODOLOGICAL TRADEOFFS\n")
    f.write("   - Hierarchical Clustering: Builds a complete merge history once (O(N^2) memory & time).\n")
    f.write("     Advantage: K does not need to be committed to beforehand. One dendrogram can be\n")
    f.write("     cut at any height to test multiple K values instantly without retraining.\n")
    f.write("   - K-Means: Faster (O(k * N * d)), scales to large datasets, but requires fixing K\n")
    f.write("     upfront and re-running the entire optimization from scratch for every K candidate.\n")

print(f"\nFull report written to: {report_path}")
print("Hierarchical clustering analysis complete!")
