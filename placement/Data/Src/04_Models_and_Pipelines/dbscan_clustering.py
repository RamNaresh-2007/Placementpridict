# Density-Based Clustering (DBSCAN)
# ==========================================================
# Every clustering method so far has forced every point into some
# cluster - K-Means (Session 21) and hierarchical clustering
# (Session 24) both partition the ENTIRE dataset, even a point that
# doesn't really belong anywhere. DBSCAN is built around a different
# idea: clusters are regions where points are packed densely
# together, and anything that isn't part of a dense region is
# explicitly labeled NOISE rather than force-fit into the nearest
# cluster. This also means DBSCAN never needs K as an input - the
# number of clusters falls out of the data's density, not a choice
# made up front.
#
# Two parameters define "dense":
#   eps: the radius (in scaled feature space) around a point to
#     search for neighbors.
#   min_samples: how many neighbors (including itself) a point needs
#     within that radius to count as a CORE point.
# Three kinds of point that result:
#   Core point: has at least min_samples neighbors within eps.
#   Border point: not a core point itself, but within eps of a core
#     point - included in that core point's cluster, but doesn't
#     extend the cluster's boundary further on its own.
#   Noise point: not a core point, and not within eps of one - not
#     assigned to ANY cluster, labeled -1.
# A cluster is then: pick an unvisited core point, gather every point
# density-reachable from it (a chain of core points within eps of each
# other, plus their border points), call that one cluster, then move
# to the next unvisited core point. Because clusters grow by chaining
# through dense regions rather than by distance to a centroid, DBSCAN
# can find clusters of arbitrary shape - not just the roughly convex,
# blob-like clusters K-Means and Ward linkage are naturally biased toward.

import sys, os

# Path configuration for reliable imports across execution contexts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "..", "05_Web_Application")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import config

os.makedirs(config.DBSCAN_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 5000  # DBSCAN on the full 35k rows was tried first and ran out of
                        # memory: at this eps scale, most points have many neighbors,
                        # and sklearn's neighbor search ends up building a very large
                        # neighbor graph across 20 dimensions. A subsample keeps the
                        # method tractable - the same tradeoff Sessions 21 and 24 made
                        # for silhouette score and hierarchical clustering respectively.

# ---------- Load, impute, scale - identical pipeline to Sessions 21, 23 & 24 ----------

train_csv_path = os.path.join(getattr(config, "SPLITS_DIR", ""), "train.csv")
if os.path.exists(train_csv_path):
    df_full = pd.read_csv(train_csv_path)
elif hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
    from sklearn.model_selection import train_test_split
    raw_df = pd.read_csv(config.RAW_DATA_PATH)
    df_full, _ = train_test_split(raw_df, test_size=0.3, random_state=config.RANDOM_STATE)
    df_full = df_full.reset_index(drop=True)
else:
    raise FileNotFoundError("Could not find train.csv in config.SPLITS_DIR or RAW_DATA_PATH.")

feature_cols = config.NUMERIC_COLUMNS
x_raw_full = df_full[feature_cols].copy()

imputer = SimpleImputer(strategy="median")
x_imputed_full = pd.DataFrame(imputer.fit_transform(x_raw_full), columns=feature_cols, index=x_raw_full.index)

scaler = StandardScaler()
x_scaled_full = scaler.fit_transform(x_imputed_full)

rng = np.random.RandomState(config.RANDOM_STATE)
sample_idx = rng.choice(len(x_scaled_full), size=SUBSAMPLE_SIZE, replace=False)
x_scaled = x_scaled_full[sample_idx]
df = df_full.iloc[sample_idx].reset_index(drop=True)

print(f"Loaded {len(df_full):,} students, subsampled {SUBSAMPLE_SIZE:,} for DBSCAN")
print(f"IsAnomaly flag present in the raw data: {df['IsAnomaly'].sum():,} students in this sample ({round(df['IsAnomaly'].mean()*100, 2)}%)")
print("(this flag was NEVER shown to DBSCAN - it's used below purely to check")
print(" whether DBSCAN's own noise detection lines up with a known anomaly label)")

# ---------- Choosing eps: the k-distance plot ----------
# min_samples is set first (a common rule of thumb: 2 x n_features,
# so 2 x 20 = 40 here, then rounded to a clean 40), then eps is read
# off a k-distance plot - for every point, find the distance to its
# k-th nearest neighbor (k = min_samples), sort all those distances,
# and look for the "elbow" where distance starts climbing sharply.
# Points to the left of the elbow are in dense regions (their k-th
# neighbor is close); points to the right are in sparse regions
# heading toward noise. The elbow is a natural boundary between the two.

MIN_SAMPLES = 40

neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)
neighbors.fit(x_scaled)
distances, _ = neighbors.kneighbors(x_scaled)
k_distances = np.sort(distances[:, -1])  # distance to the k-th (farthest) neighbor, sorted ascending

# Numeric elbow estimate: a straight max-second-difference approach
# (as Session 21 used for the K-Means elbow) turned out to be too
# sensitive to noise at the very tail of this curve, picking a point
# almost at the maximum distance - which produced eps values so large
# that everything merged into one cluster with zero noise, defeating
# the point of the demonstration. The "kneedle" approach used instead
# is more robust for a curve shaped like this one (long flat run, then
# a rise): normalize both axes to 0-1, draw a straight line between
# the curve's first and last point, and pick whichever point sits
# FARTHEST from that line (perpendicular distance) - the point where
# the curve bends away from a straight line the most.
n_points = len(k_distances)
x_norm = np.linspace(0, 1, n_points)
y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min())
x1, y1 = x_norm[0], y_norm[0]
x2, y2 = x_norm[-1], y_norm[-1]
numerator = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
perpendicular_distance = numerator / denominator
elbow_idx = int(np.argmax(perpendicular_distance))
suggested_eps = round(float(k_distances[elbow_idx]), 2)
print(f"Suggested eps from k-distance knee (at the {round(elbow_idx/n_points*100, 1)}th percentile of sorted distances): {suggested_eps}")

plt.figure(figsize=(9, 5))
plt.plot(k_distances, color="#6E2436", linewidth=2, label=f"k-distances (k={MIN_SAMPLES})")
plt.axvline(x=elbow_idx, color="#2B5B84", linestyle="--", alpha=0.75,
            label=f"Knee index ({elbow_idx:,} / {n_points:,} = {round(elbow_idx/n_points*100, 1)}%)")
plt.axhline(y=suggested_eps, color="#D9534F", linestyle="--", alpha=0.75,
            label=f"Suggested eps = {suggested_eps}")
plt.scatter([elbow_idx], [suggested_eps], color="#D9534F", s=65, zorder=5)
plt.xlabel("Points, sorted by distance to their k-th nearest neighbor", fontsize=11)
plt.ylabel(f"Distance to {MIN_SAMPLES}-th nearest neighbor", fontsize=11)
plt.title(f"K-Distance Plot (min_samples={MIN_SAMPLES}) - Knee at eps={suggested_eps}", fontsize=12, fontweight="bold")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{config.DBSCAN_DIR}/k_distance_plot.png", dpi=300)
plt.close()
print(f"K-distance plot saved (min_samples={MIN_SAMPLES})")

# ---------- Fit DBSCAN at the suggested eps ----------

EPS = suggested_eps
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=1)
cluster_labels = dbscan.fit_predict(x_scaled)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = int((cluster_labels == -1).sum())
n_core = len(dbscan.core_sample_indices_)

print(f"\nDBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}):")
print(f"  Clusters found: {n_clusters}")
print(f"  Core points:    {n_core:,} ({round(n_core/len(df)*100, 1)}%)")
print(f"  Noise points:   {n_noise:,} ({round(n_noise/len(df)*100, 1)}%)")

cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("\nCluster sizes (-1 = noise):")
print(cluster_sizes.to_string())

# ---------- Does DBSCAN's noise line up with the dataset's own anomaly flag? ----------
# This is the check the IsAnomaly column exists to make possible -
# DBSCAN never saw it, so any overlap is a genuine finding, not
# something DBSCAN was tuned toward.

is_noise = (cluster_labels == -1)
anomaly_flag = df["IsAnomaly"].values.astype(bool)
both_flagged = int((is_noise & anomaly_flag).sum())
noise_that_is_anomaly = round(both_flagged / max(n_noise, 1) * 100, 1)
anomaly_that_is_noise = round(both_flagged / max(anomaly_flag.sum(), 1) * 100, 1)

print(f"\nOverlap between DBSCAN noise and the dataset's IsAnomaly flag:")
print(f"  Points flagged as BOTH noise and anomalous: {both_flagged:,}")
print(f"  Of DBSCAN's noise points, {noise_that_is_anomaly}% are also flagged IsAnomaly=1")
print(f"  Of IsAnomaly=1 students, {anomaly_that_is_noise}% were caught as DBSCAN noise")

# ---------- eps sensitivity: too small vs too large ----------
# A quick before/after showing why eps needs to be chosen carefully -
# too small fragments the data into mostly-noise; too large merges
# everything into one giant cluster and stops finding noise at all.

eps_values_to_try = [round(EPS * 0.5, 2), EPS, round(EPS * 2, 2)]
sensitivity_results = []
print("\neps sensitivity (min_samples held fixed):")
for eps_val in eps_values_to_try:
    db = DBSCAN(eps=eps_val, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=1)
    labels = db.fit_predict(x_scaled)
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    n_ns = int((labels == -1).sum())
    sensitivity_results.append({"eps": eps_val, "n_clusters": n_clust, "n_noise": n_ns})
    print(f"  eps={eps_val}: {n_clust} clusters, {n_ns:,} noise points ({round(n_ns/len(df)*100,1)}%)")

# ---------- Visualize: PCA down to 2D, noise vs clusters ----------
# Same PCA-for-plotting-only approach as Session 21 - clustering
# itself used all 20 dimensions, this compresses to 2D purely to see it.

pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
x_2d = pca.fit_transform(x_scaled)

plt.figure(figsize=(9, 7))
noise_mask = cluster_labels == -1
plt.scatter(x_2d[~noise_mask, 0], x_2d[~noise_mask, 1], c=cluster_labels[~noise_mask],
            cmap="tab10", s=8, alpha=0.5, label="Clustered")
plt.scatter(x_2d[noise_mask, 0], x_2d[noise_mask, 1], c="black", marker="x", s=22,
            alpha=0.8, label=f"Noise ({n_noise:,} points)")
plt.xlabel(f"PCA Component 1 ({round(pca.explained_variance_ratio_[0]*100, 1)}% variance)", fontsize=11)
plt.ylabel(f"PCA Component 2 ({round(pca.explained_variance_ratio_[1]*100, 1)}% variance)", fontsize=11)
plt.title(f"DBSCAN Clusters vs Noise, Visualized in 2D via PCA (eps={EPS}, min_samples={MIN_SAMPLES})",
          fontsize=12, fontweight="bold")
plt.legend(loc="upper right", frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{config.DBSCAN_DIR}/dbscan_clusters_pca_2d.png", dpi=300)
plt.close()
print("\n2D PCA visualization (clusters vs noise) saved")

# ---------- Placement rate: noise points vs clustered points ----------
# Same after-the-fact check as Sessions 21 and 24 - the label was
# never used to fit DBSCAN.

placement_clustered = df.loc[~noise_mask, config.TARGET_COLUMN].mean() if (~noise_mask).any() else float("nan")
placement_noise = df.loc[noise_mask, config.TARGET_COLUMN].mean() if noise_mask.any() else float("nan")
print(f"\nPlacement rate, clustered points:  {round(placement_clustered*100, 1) if not np.isnan(placement_clustered) else 'n/a'}%")
print(f"Placement rate, noise points:      {round(placement_noise*100, 1) if not np.isnan(placement_noise) else 'n/a (no noise points found)'}%")

# ---------- Save report ----------

report_path = os.path.join(config.DBSCAN_DIR, "dbscan_report.txt")
with open(report_path, "w") as f:
    f.write(f"DBSCAN on {len(df):,} students, {len(feature_cols)} scaled, median-imputed\n")
    f.write("numeric features (same pipeline as Sessions 21, 23 & 24).\n\n")
    f.write(f"min_samples = {MIN_SAMPLES} (rule of thumb: 2 x n_features)\n")
    f.write(f"eps = {EPS} (read from k-distance plot elbow, max-curvature method)\n\n")
    f.write(f"Clusters found: {n_clusters}\n")
    f.write(f"Core points:    {n_core:,} ({round(n_core/len(df)*100, 1)}%)\n")
    f.write(f"Noise points:   {n_noise:,} ({round(n_noise/len(df)*100, 1)}%)\n\n")
    f.write("Cluster sizes (-1 = noise):\n")
    f.write(cluster_sizes.to_string())
    f.write(f"\n\nOverlap with the dataset's own IsAnomaly flag (never shown to DBSCAN):\n")
    f.write(f"  Points flagged as both noise and anomalous: {both_flagged:,}\n")
    f.write(f"  Of DBSCAN's noise points, {noise_that_is_anomaly}% are also IsAnomaly=1\n")
    f.write(f"  Of IsAnomaly=1 students, {anomaly_that_is_noise}% were caught as noise\n\n")
    f.write("eps sensitivity (min_samples held fixed):\n")
    for r in sensitivity_results:
        f.write(f"  eps={r['eps']}: {r['n_clusters']} clusters, {r['n_noise']:,} noise points\n")
    f.write(f"\nPlacement rate, clustered points: {round(placement_clustered*100, 1) if not np.isnan(placement_clustered) else 'n/a'}%\n")
    f.write(f"Placement rate, noise points:     {round(placement_noise*100, 1) if not np.isnan(placement_noise) else 'n/a (no noise points found)'}%\n")
    f.write("\nDBSCAN vs Sessions 21/24:\n")
    f.write("  K-Means and Ward linkage force every point into a cluster and need\n")
    f.write("  K chosen up front. DBSCAN infers cluster count from density and\n")
    f.write("  explicitly separates out points that don't belong anywhere -\n")
    f.write("  at the cost of needing eps and min_samples tuned instead of K,\n")
    f.write("  and struggling when clusters have very different densities.\n")

print(f"\nReport saved to {report_path}")
