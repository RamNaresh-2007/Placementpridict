
import os
import pandas as pd
import matplotlib.pyplot as plt




import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "01_Datasets",
    "placement_predict_50k Dataset (2).csv"
)

df = pd.read_csv(DATA_PATH)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("=" * 60)

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# IDENTIFY CATEGORIES
# ==========================================================

print("\n" + "=" * 60)
print("CATEGORICAL DATA / CATEGORIES")
print("=" * 60)

# Identify categorical columns
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

# Display unique categories in each categorical column
for column in categorical_columns:
    print("\n" + "-" * 50)
    print("Column:", column)
    print("Number of Categories:", df[column].nunique())
    print("Categories:")
    print(df[column].dropna().unique())

    print("\nCategory Counts:")
    print(df[column].value_counts())

# ----------------------------------------------------------
# Numerical Columns
# ----------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\n" + "=" * 60)
print("NUMERICAL COLUMNS")
print("=" * 60)

print(numeric_columns)

# ----------------------------------------------------------
# Column-wise Data Type
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ----------------------------------------------------------
# Create Plots Folder
# ----------------------------------------------------------

PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ==========================================================
# UNIVARIATE ANALYSIS
# ==========================================================

# Histogram
if "CGPA" in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df["CGPA"], bins=10, edgecolor="black")
    plt.title("Distribution of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "Histogram_CGPA.png")
    )
    plt.close()

# ----------------------------------------------------------
# Box Plot
# ----------------------------------------------------------

if "CGPA" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df["CGPA"])
    plt.title("Box Plot of CGPA")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "BoxPlot_CGPA.png")
    )
    plt.close()

# ----------------------------------------------------------
# Density Plot (KDE)
# ----------------------------------------------------------

if "CGPA" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df["CGPA"], fill=True)
    plt.title("Density Plot of CGPA")
    plt.xlabel("CGPA")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "Density_CGPA.png")
    )
    plt.close()

# ----------------------------------------------------------
# Violin Plot
# ----------------------------------------------------------

if "CGPA" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.violinplot(y=df["CGPA"])
    plt.title("Violin Plot of CGPA")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "Violin_CGPA.png")
    )
    plt.close()

# ----------------------------------------------------------
# Count Plot (Placement Status)
# ----------------------------------------------------------

if "PlacementStatus" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="PlacementStatus")
    plt.title("Placement Status Distribution")
    plt.xlabel("Placement Status")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "Placement_CountPlot.png")
    )
    plt.close()

# ----------------------------------------------------------
# Pie Chart (Gender)
# ----------------------------------------------------------

if "Gender" in df.columns:
    plt.figure(figsize=(6, 6))

    df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "Gender_PieChart.png")
    )
    plt.close()

# ==========================================================
# BIVARIATE ANALYSIS
# ==========================================================

# Scatter Plot
if (
    "CGPA" in df.columns
    and "AttendancePercent" in df.columns
):
    plt.figure(figsize=(6, 4))

    sns.scatterplot(
        x="CGPA",
        y="AttendancePercent",
        data=df
    )

    plt.title("CGPA vs AttendancePercent")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Scatter_CGPA_Attendance.png"
        )
    )
    plt.close()

# ----------------------------------------------------------
# Regression Plot
# ----------------------------------------------------------

if (
    "CGPA" in df.columns
    and "AttendancePercent" in df.columns
):
    plt.figure(figsize=(6, 4))

    sns.regplot(
        x="CGPA",
        y="AttendancePercent",
        data=df
    )

    plt.title("Regression Plot")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Regression_CGPA_Attendance.png"
        )
    )
    plt.close()

# ----------------------------------------------------------
# Box Plot (CGPA vs Placement Status)
# ----------------------------------------------------------

if (
    "PlacementStatus" in df.columns
    and "CGPA" in df.columns
):
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        x="PlacementStatus",
        y="CGPA",
        data=df
    )

    plt.title("CGPA vs Placement Status")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "CGPA_vs_Placement_BoxPlot.png"
        )
    )
    plt.close()

# ----------------------------------------------------------
# Violin Plot (CGPA vs Placement Status)
# ----------------------------------------------------------

if (
    "PlacementStatus" in df.columns
    and "CGPA" in df.columns
):
    plt.figure(figsize=(6, 4))

    sns.violinplot(
        x="PlacementStatus",
        y="CGPA",
        data=df
    )

    plt.title("CGPA vs Placement Status")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "CGPA_vs_Placement_Violin.png"
        )
    )
    plt.close()

# ----------------------------------------------------------
# Count Plot (Gender vs Placement Status)
# ----------------------------------------------------------

if (
    "Gender" in df.columns
    and "PlacementStatus" in df.columns
):
    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="Gender",
        hue="PlacementStatus",
        data=df
    )

    plt.title("Gender vs Placement Status")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Gender_vs_Placement_CountPlot.png"
        )
    )
    plt.close()

# ----------------------------------------------------------
# Stacked Bar Chart
# ----------------------------------------------------------

if (
    "Gender" in df.columns
    and "PlacementStatus" in df.columns
):
    cross_table = pd.crosstab(
        df["Gender"],
        df["PlacementStatus"]
    )

    cross_table.plot(
        kind="bar",
        stacked=True,
        figsize=(6, 4)
    )

    plt.title("Gender vs Placement Status")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Gender_vs_Placement_StackedBar.png"
        )
    )
    plt.close()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

if not numeric_df.empty:

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Correlation_Heatmap.png"
        )
    )
    plt.close()

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("Plots saved in:", PLOT_DIR)
print("=" * 60)

