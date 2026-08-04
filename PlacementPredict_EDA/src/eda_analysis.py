"""
PlacementPredict - Exploratory Data Analysis
=============================================
Runs a complete EDA pipeline on the student placement dataset and saves
every plot and text report to the outputs/ folder.

Run this file directly (PyCharm: right-click -> Run, or `python src/eda_analysis.py`
from the project root).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write plots to disk without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
# BASE_DIR = project root (one level above this src/ file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary_report.txt")
report_lines = []  # collects every printed section so it can also be saved to a .txt file


def log(text=""):
    """Print to console AND remember the line so it lands in the saved report."""
    print(text)
    report_lines.append(str(text))


def section(title):
    log("\n" + "=" * 80)
    log(title)
    log("=" * 80)


def save_plot(fig, filename):
    """Save a figure into outputs/plots and close it to free memory."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    log(f"[saved plot] {filename}")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
section("1. LOADING DATA")
df = pd.read_csv(DATA_PATH)
log(f"Dataset loaded from: {DATA_PATH}")
log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Column groups used throughout the script.
# Splitting columns like this up front keeps every later step simple and explicit
# instead of re-detecting types again and again.
categorical_cols = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier",
]
binary_target_cols = ["PlacementStatus", "IsAnomaly"]
numeric_cols = [c for c in df.columns
                if c not in categorical_cols + binary_target_cols + ["StudentID"]]

# ---------------------------------------------------------------------------
# 2. First look at the data
# ---------------------------------------------------------------------------
section("2. FIRST LOOK")
log("First 5 rows:")
log(df.head().to_string())

log("\nColumn data types:")
log(df.dtypes.to_string())

# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------
section("3. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0].sort_values(
    "missing_count", ascending=False
)
log(missing_report.to_string() if not missing_report.empty else "No missing values found.")

if not missing_report.empty:
    fig, ax = plt.subplots()
    sns.barplot(x=missing_report.index, y=missing_report["missing_percent"], ax=ax, color="indianred")
    ax.set_title("Missing Values by Column (%)")
    ax.set_ylabel("Percent missing")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, "01_missing_values.png")

# ---------------------------------------------------------------------------
# 4. Duplicate rows
# ---------------------------------------------------------------------------
section("4. DUPLICATE ROWS")
dup_count = df.duplicated().sum()
log(f"Fully duplicated rows: {dup_count}")
dup_id_count = df.duplicated(subset=["StudentID"]).sum()
log(f"Duplicate StudentID values: {dup_id_count}")

# ---------------------------------------------------------------------------
# 5. Descriptive statistics (numeric columns)
# ---------------------------------------------------------------------------
section("5. DESCRIPTIVE STATISTICS - NUMERIC COLUMNS")
desc = df[numeric_cols].describe().T
desc["skew"] = df[numeric_cols].skew()
desc["missing"] = df[numeric_cols].isnull().sum()
log(desc.round(2).to_string())

# ---------------------------------------------------------------------------
# 6. Categorical column frequencies
# ---------------------------------------------------------------------------
section("6. CATEGORICAL COLUMN FREQUENCIES")
for col in categorical_cols:
    log(f"\n--- {col} ---")
    log(df[col].value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# 7. Target variable balance (PlacementStatus)
# ---------------------------------------------------------------------------
section("7. TARGET VARIABLE - PLACEMENT STATUS")
placement_counts = df["PlacementStatus"].value_counts()
placement_pct = df["PlacementStatus"].value_counts(normalize=True).mul(100).round(2)
log("Counts:")
log(placement_counts.to_string())
log("\nPercentage:")
log(placement_pct.to_string())

fig, ax = plt.subplots()
sns.countplot(x="PlacementStatus", data=df, ax=ax, hue="PlacementStatus", palette=["#e74c3c", "#2ecc71"], legend=False)
ax.set_title("Placement Status Distribution (0 = Not Placed, 1 = Placed)")
ax.set_xlabel("Placement Status")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
save_plot(fig, "02_placement_status_distribution.png")

# ---------------------------------------------------------------------------
# 8. Numeric feature distributions
# ---------------------------------------------------------------------------
section("8. NUMERIC FEATURE DISTRIBUTIONS")
dist_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore",
             "CodingTestScore", "MockInterviewScore", "SoftSkillsRating"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col in zip(axes.flatten(), dist_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
fig.tight_layout()
save_plot(fig, "03_numeric_distributions.png")
log("Plotted histograms for: " + ", ".join(dist_cols))

# ---------------------------------------------------------------------------
# 9. Outlier detection (IQR method)
# ---------------------------------------------------------------------------
section("9. OUTLIER DETECTION (IQR METHOD)")
outlier_summary = {}
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = n_outliers
outlier_df = pd.Series(outlier_summary, name="outlier_count").sort_values(ascending=False)
log(outlier_df.to_string())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
box_cols = ["CGPA", "AttendancePercent", "CodingTestScore",
            "AptitudeTestScore", "MockInterviewScore", "Salary Package"]
for ax, col in zip(axes.flatten(), box_cols):
    sns.boxplot(y=df[col], ax=ax, color="lightsalmon")
    ax.set_title(f"Boxplot - {col}")
fig.tight_layout()
save_plot(fig, "04_outlier_boxplots.png")

# ---------------------------------------------------------------------------
# 10. Correlation analysis
# ---------------------------------------------------------------------------
section("10. CORRELATION ANALYSIS")
corr = df[numeric_cols + ["PlacementStatus"]].corr(numeric_only=True)
log("Top correlations with PlacementStatus:")
log(corr["PlacementStatus"].drop("PlacementStatus").sort_values(ascending=False).round(3).to_string())

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
ax.set_title("Correlation Heatmap - Numeric Features")
save_plot(fig, "05_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 11. Categorical features vs Placement Status
# ---------------------------------------------------------------------------
section("11. CATEGORICAL FEATURES VS PLACEMENT STATUS")
cat_vs_target_cols = ["CollegeTier", "Stream", "Specialisation", "Hostel", "CGPA_Tier"]
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, col in zip(axes.flatten(), cat_vs_target_cols):
    placement_rate = df.groupby(col)["PlacementStatus"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=placement_rate.index, y=placement_rate.values, ax=ax, color="mediumseagreen")
    ax.set_title(f"Placement Rate (%) by {col}")
    ax.set_ylabel("Placement Rate (%)")
    ax.tick_params(axis="x", rotation=45)
axes.flatten()[-1].axis("off")  # unused subplot slot
fig.tight_layout()
save_plot(fig, "06_placement_rate_by_category.png")

for col in cat_vs_target_cols:
    rate = df.groupby(col)["PlacementStatus"].mean().mul(100).round(2)
    log(f"\nPlacement rate (%) by {col}:")
    log(rate.sort_values(ascending=False).to_string())

# ---------------------------------------------------------------------------
# 12. CGPA / Attendance vs Placement (boxplots)
# ---------------------------------------------------------------------------
section("12. KEY NUMERIC FEATURES VS PLACEMENT STATUS")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col in zip(axes, ["CGPA", "AttendancePercent", "CodingTestScore"]):
    sns.boxplot(x="PlacementStatus", y=col, data=df, ax=ax, hue="PlacementStatus", palette=["#e74c3c", "#2ecc71"], legend=False)
    ax.set_title(f"{col} vs Placement Status")
fig.tight_layout()
save_plot(fig, "07_numeric_vs_placement_boxplots.png")

# ---------------------------------------------------------------------------
# 13. Pairwise relationships (sampled for readability)
# ---------------------------------------------------------------------------
section("13. PAIRWISE RELATIONSHIPS (SAMPLE)")
sample_df = df.sample(n=min(2000, len(df)), random_state=42)
pair_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
pair_plot = sns.pairplot(sample_df[pair_cols], hue="PlacementStatus",
                          palette=["#e74c3c", "#2ecc71"], plot_kws={"alpha": 0.5, "s": 15})
pair_plot.fig.suptitle("Pairwise Relationships (2000-row sample)", y=1.02)
pair_plot.savefig(os.path.join(PLOTS_DIR, "08_pairplot.png"), bbox_inches="tight", dpi=120)
plt.close(pair_plot.fig)
log("[saved plot] 08_pairplot.png")

# ---------------------------------------------------------------------------
# 14. Salary Package analysis (placed students only)
# ---------------------------------------------------------------------------
section("14. SALARY PACKAGE ANALYSIS (PLACED STUDENTS)")
placed_df = df[df["PlacementStatus"] == 1]
log(placed_df["Salary Package"].describe().round(2).to_string())

fig, ax = plt.subplots()
sns.histplot(placed_df["Salary Package"], kde=True, ax=ax, color="darkorange")
ax.set_title("Salary Package Distribution (Placed Students Only)")
save_plot(fig, "09_salary_distribution.png")

fig, ax = plt.subplots()
sns.boxplot(x="CollegeTier", y="Salary Package", data=placed_df, ax=ax,
            order=["Tier1", "Tier2", "Tier3"], color="gold")
ax.set_title("Salary Package by College Tier (Placed Students)")
save_plot(fig, "10_salary_by_college_tier.png")

# ---------------------------------------------------------------------------
# 15. Anomaly flag overview
# ---------------------------------------------------------------------------
section("15. ANOMALY FLAG OVERVIEW")
anomaly_counts = df["IsAnomaly"].value_counts()
log(anomaly_counts.to_string())
log(f"Anomaly rate: {df['IsAnomaly'].mean() * 100:.2f}%")

# ---------------------------------------------------------------------------
# 16. SGPA trend across semesters
# ---------------------------------------------------------------------------
section("16. AVERAGE SGPA TREND ACROSS SEMESTERS")
sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
sgpa_by_placement = df.groupby("PlacementStatus")[sgpa_cols].mean()
log(sgpa_by_placement.round(2).to_string())

fig, ax = plt.subplots()
for status, label, color in [(0, "Not Placed", "#e74c3c"), (1, "Placed", "#2ecc71")]:
    ax.plot(range(1, 9), sgpa_by_placement.loc[status], marker="o", label=label, color=color)
ax.set_xlabel("Semester")
ax.set_ylabel("Average SGPA")
ax.set_title("Average SGPA Trend by Placement Status")
ax.legend()
save_plot(fig, "11_sgpa_trend.png")

# ---------------------------------------------------------------------------
# 17. Save full text report
# ---------------------------------------------------------------------------
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

section("EDA COMPLETE")
log(f"Text report saved to : {REPORT_PATH}")
log(f"Plots saved to       : {PLOTS_DIR}")
log(f"Total plots generated: {len(os.listdir(PLOTS_DIR))}")
