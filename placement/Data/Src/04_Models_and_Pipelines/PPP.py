# ============================================================
# PLACEMENT PREDICTION PROJECT
# EDA -> CLEANING -> ENCODING -> FEATURE SCALING
# MODEL-READY DATA
# ============================================================
#
# Dataset:
# C:\Users\ADMIN\PycharmProjects\PPProgram\
# placement_predict_50k Dataset.csv
#
# IMPORTANT:
# - No plt.show()
# - Plots are automatically saved
# - Only important EDA plots are generated
# - No plot windows will open
# - Train/Test split happens BEFORE preprocessing
# - Preprocessing is FIT only on training data
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

# Prevent plots from opening
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)

import joblib

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "01_Datasets",
    "placement_predict_50k Dataset (2).csv"
)

EDA_FOLDER = os.path.join(
    BASE_DIR,
    "EDA_Outputs"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "processed_data"
)

os.makedirs(
    EDA_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 80)
print("PLACEMENT PREDICTION PROJECT")
print("=" * 80)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "\nDataset not found at:\n"
        + DATA_PATH
        + "\n\n"
        "Make sure placement_predict_50k Dataset (2).csv "
        "is in the same folder as PPP.py."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 4. BASIC DATASET INSPECTION
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC DATASET INSPECTION")
print("=" * 80)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")

for i, column in enumerate(df.columns, start=1):
    print(i, "->", column)

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# ============================================================
# 5. NUMERICAL / CATEGORICAL FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=[
        "object",
        "category",
        "string"
    ]
).columns.tolist()


print("\n" + "=" * 80)
print("2. FEATURE TYPES")
print("=" * 80)

print("\nNumerical columns:")

for column in numeric_columns:
    print("-", column)

print("\nCategorical columns:")

for column in categorical_columns:
    print("-", column)

print(
    "\nNumber of numerical columns:",
    len(numeric_columns)
)

print(
    "Number of categorical columns:",
    len(categorical_columns)
)


# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("3. STATISTICAL SUMMARY")
print("=" * 80)

print("\nNumerical summary:")

numerical_summary = (
    df
.select_dtypes(include=np.number)
    .describe()
    .transpose()
)

print(numerical_summary)

numerical_summary.to_csv(
    os.path.join(
        EDA_FOLDER,
        "01_numerical_summary.csv"
    )
)


if len(categorical_columns) > 0:

    categorical_summary = (
        df[categorical_columns]
        .describe()
        .transpose()
    )

    print("\nCategorical summary:")
    print(categorical_summary)

    categorical_summary.to_csv(
        os.path.join(
            EDA_FOLDER,
            "02_categorical_summary.csv"
        )
    )


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. MISSING VALUE ANALYSIS")
print("=" * 80)

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum()
    / len(df)
    * 100
)

missing_table = pd.DataFrame({

    "Feature": df.columns,

    "Missing_Count":
        missing_count.values,

    "Missing_Percentage":
        missing_percentage.values
})

missing_table = missing_table.sort_values(
    "Missing_Count",
    ascending=False
)

print(missing_table)

missing_table.to_csv(
    os.path.join(
        EDA_FOLDER,
        "03_missing_values.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Missing-value plot
# Only one plot
# ------------------------------------------------------------

missing_plot = missing_table[
    missing_table["Missing_Count"] > 0
]

if len(missing_plot) > 0:

    fig = plt.figure(figsize=(10, 6))

    sns.barplot(
        data=missing_plot,
        x="Missing_Count",
        y="Feature"
    )

    plt.title("Missing Values by Feature")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "04_missing_values.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

else:

    print("\nNo missing values found.")


# ============================================================
# 8. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_count
)

if duplicate_count > 0:

    print("\nRemoving duplicate rows...")

    df = df.drop_duplicates()

    print(
        "Shape after removing duplicates:",
        df.shape
    )

else:

    print("No duplicate rows found.")


# ============================================================
# 9. TARGET VARIABLE
# ============================================================

TARGET = "PlacementStatus"

if TARGET not in df.columns:

    raise ValueError(
        "PlacementStatus column not found."
    )


print("\n" + "=" * 80)
print("6. TARGET VARIABLE")
print("=" * 80)

print(
    "\nTarget:",
    TARGET
)

print("\nTarget values:")
print(
    df[TARGET].value_counts()
)

target_percentage = (
    df[TARGET]
    .value_counts(normalize=True)
    * 100
).round(2)

print("\nTarget percentages:")
print(target_percentage)


target_summary = pd.DataFrame({

    "Count":
        df[TARGET].value_counts(),

    "Percentage":
        target_percentage

})

target_summary.to_csv(
    os.path.join(
        EDA_FOLDER,
        "05_target_distribution.csv"
    )
)


# ------------------------------------------------------------
# TARGET PLOT
# ------------------------------------------------------------

fig = plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=TARGET
)

plt.title(
    "Placement Status Distribution"
)

plt.xlabel(
    "Placement Status"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_FOLDER,
        "06_target_distribution.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("7. UNIVARIATE ANALYSIS")
print("=" * 80)

print(
    """
Univariate analysis studies ONE variable at a time.

We will use:
1. Distribution
2. Histogram
3. Boxplot
4. Category frequency
"""
)


# ------------------------------------------------------------
# Select IMPORTANT numerical features
# ------------------------------------------------------------

important_numeric = [

    "CGPA",

    "AttendancePercent",

    "AptitudeTestScore",

    "CodingTestScore",

    "MockInterviewScore",

    "Internships",

    "Projects"

]


important_numeric = [

    column

    for column in important_numeric

    if column in df.columns

]


print(
    "\nImportant numerical features:"
)

for column in important_numeric:
    print("-", column)


# ============================================================
# 10. COMBINED NUMERICAL DISTRIBUTION PLOT
# ============================================================

print(
    "\nCreating numerical distribution plot..."
)

if len(important_numeric) > 0:

    # One figure containing selected distributions
    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 8)
    )

    axes = axes.flatten()

    for i, column in enumerate(
        important_numeric
    ):

        sns.histplot(
            df[column],
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(
            column
        )

        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")

    # Hide unused axes

    for j in range(
        len(important_numeric),
        len(axes)
    ):

        axes[j].set_visible(False)

    plt.suptitle(
        "Univariate Analysis - Important Numerical Features",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "07_univariate_distributions.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 11. COMBINED BOXPLOT
# ============================================================

print(
    "\nCreating numerical boxplot..."
)

if len(important_numeric) > 0:

    # Convert to long format

    boxplot_df = df[
        important_numeric
    ].melt(
        var_name="Feature",
        value_name="Value"
    )

    fig = plt.figure(
        figsize=(12, 7)
    )

    sns.boxplot(
        data=boxplot_df,
        x="Feature",
        y="Value"
    )

    plt.title(
        "Univariate Analysis - Outlier Distribution"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "08_univariate_boxplots.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 12. CATEGORICAL SUMMARY
# ============================================================

print(
    "\nCategorical feature frequencies:"
)

categorical_summary_list = []

for column in categorical_columns:

    if column == TARGET:
        continue

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
        .head(10)
    )

    for category, count in counts.items():

        categorical_summary_list.append({

            "Feature":
                column,

            "Category":
                category,

            "Count":
                count

        })


if len(categorical_summary_list) > 0:

    categorical_frequency_df = pd.DataFrame(
        categorical_summary_list
    )

    print(
        categorical_frequency_df
    )

    categorical_frequency_df.to_csv(
        os.path.join(
            EDA_FOLDER,
            "09_categorical_frequency.csv"
        ),
        index=False
    )


# ============================================================
# ============================================================
#                  BIVARIATE ANALYSIS
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("8. BIVARIATE ANALYSIS")
print("=" * 80)

print(
    """
Bivariate analysis studies the relationship
between TWO variables.

We will examine:

1. Numerical feature vs PlacementStatus
2. Categorical feature vs PlacementStatus
3. Numerical feature vs Numerical feature
"""
)


# ============================================================
# 13. NUMERICAL VS TARGET
# ============================================================

print(
    "\nCreating numerical vs target plots..."
)

target_box_features = [

    "CGPA",

    "AttendancePercent",

    "AptitudeTestScore",

    "CodingTestScore",

    "MockInterviewScore"

]


target_box_features = [

    column

    for column in target_box_features

    if column in df.columns

]


if len(target_box_features) > 0:

    fig, axes = plt.subplots(

        nrows=2,

        ncols=3,

        figsize=(15, 9)

    )

    axes = axes.flatten()

    for i, column in enumerate(
        target_box_features
    ):

        sns.boxplot(

            data=df,

            x=TARGET,

            y=column,

            ax=axes[i]

        )

        axes[i].set_title(
            f"{column} vs {TARGET}"
        )

        axes[i].set_xlabel(
            TARGET
        )

        axes[i].set_ylabel(
            column
        )

    # Hide unused subplot

    for j in range(
        len(target_box_features),
        len(axes)
    ):

        axes[j].set_visible(False)

    plt.suptitle(
        "Bivariate Analysis - Numerical Features vs Placement",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "10_numerical_vs_target.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 14. CATEGORICAL VS TARGET
# ============================================================

print(
    "\nCategorical vs target analysis..."
)

# Select useful categorical columns
# instead of plotting every categorical variable.

preferred_categorical = [

    "Gender",

    "CollegeTier",

    "Degree",

    "Branch",

    "Region"

]


preferred_categorical = [

    column

    for column in preferred_categorical

    if column in df.columns

]


if len(preferred_categorical) > 0:

    fig, axes = plt.subplots(

        nrows=2,

        ncols=3,

        figsize=(16, 9)

    )

    axes = axes.flatten()

    for i, column in enumerate(
        preferred_categorical
    ):

        # Top categories only

        top_categories = (

            df[column]
            .value_counts()
            .head(10)
            .index

        )

        plot_df = df[
            df[column]
            .isin(top_categories)
        ]

        sns.countplot(

            data=plot_df,

            x=column,

            hue=TARGET,

            ax=axes[i]

        )

        axes[i].set_title(
            f"{column} vs {TARGET}"
        )

        axes[i].tick_params(
            axis="x",
            rotation=45
        )

    for j in range(
        len(preferred_categorical),
        len(axes)
    ):

        axes[j].set_visible(False)

    plt.suptitle(
        "Bivariate Analysis - Categorical Features vs Placement",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "11_categorical_vs_target.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 15. NUMERICAL VS NUMERICAL
# ============================================================

print(
    "\nCreating numerical vs numerical scatter plot..."
)

if (
    "CGPA" in df.columns
    and
    "CodingTestScore" in df.columns
):

    fig = plt.figure(
        figsize=(9, 6)
    )
    sns.scatterplot(

        data=df.sample(
            min(5000, len(df)),
            random_state=42
        ),

        x="CGPA",

        y="CodingTestScore",

        hue=TARGET,

        alpha=0.5

    )

    plt.title(
        "CGPA vs Coding Test Score"
    )

    plt.xlabel(
        "CGPA"
    )

    plt.ylabel(
        "Coding Test Score"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "12_numeric_vs_numeric.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ============================================================
#                  MULTIVARIATE ANALYSIS
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("9. MULTIVARIATE ANALYSIS")
print("=" * 80)

print(
    """
Multivariate analysis examines
THREE OR MORE variables together.

We will use:

1. Correlation heatmap
2. Selected-feature pairplot
"""
)


# ============================================================
# 16. CORRELATION HEATMAP
# ============================================================

print(
    "\nCreating correlation heatmap..."
)

if len(important_numeric) >= 2:

    correlation_matrix = (

        df[
            important_numeric
        ]
        .corr()

    )

    correlation_matrix.to_csv(
        os.path.join(
            EDA_FOLDER,
            "13_correlation_matrix.csv"
        )
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(

        correlation_matrix,

        annot=True,

        fmt=".2f",

        cmap="coolwarm",

        linewidths=0.3

    )

    plt.title(
        "Correlation Heatmap - Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "14_correlation_heatmap.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 17. PAIRPLOT
# ============================================================

print(
    "\nCreating selected-feature pairplot..."
)

pair_features = [

    "CGPA",

    "AptitudeTestScore",

    "CodingTestScore",

    "MockInterviewScore"

]


pair_features = [

    column

    for column in pair_features

    if column in df.columns

]


if len(pair_features) >= 2:

    pair_df = df[
        pair_features + [TARGET]
    ].dropna()

    # 50K rows is too much for pairplot.
    # Use a representative sample.

    if len(pair_df) > 2000:

        pair_df = pair_df.sample(
            2000,
            random_state=42
        )

    pair_grid = sns.pairplot(

        pair_df,

        hue=TARGET,

        diag_kind="hist"

    )

    pair_grid.fig.suptitle(
        "Multivariate Pairplot - Selected Placement Features",
        y=1.02
    )

    pair_grid.fig.savefig(
        os.path.join(
            EDA_FOLDER,
            "15_pairplot.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        pair_grid.fig
    )


# ============================================================
# ============================================================
#                    OUTLIER ANALYSIS
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("10. OUTLIER ANALYSIS")
print("=" * 80)


outlier_results = []


for column in important_numeric:

    Q1 = df[column].quantile(
        0.25
    )

    Q3 = df[column].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower_fence = (
        Q1 - 1.5 * IQR
    )

    upper_fence = (
        Q3 + 1.5 * IQR
    )

    outlier_count = (

        (
            (df[column] < lower_fence)
            |
            (df[column] > upper_fence)
        )

        .sum()

    )

    outlier_percentage = (

        outlier_count
        /
        len(df)
        *
        100

    )
    outlier_results.append({

        "Feature":
            column,

        "Q1":
            Q1,

        "Q3":
            Q3,

        "IQR":
            IQR,

        "Lower_Fence":
            lower_fence,

        "Upper_Fence":
            upper_fence,

        "Outlier_Count":
            outlier_count,

        "Outlier_Percentage":
            outlier_percentage

    })


outlier_table = pd.DataFrame(
    outlier_results
)

print(
    outlier_table
)

outlier_table.to_csv(
    os.path.join(
        EDA_FOLDER,
        "16_outlier_analysis.csv"
    ),
    index=False
)


# ============================================================
# 18. SAVE CLEANED EDA DATA
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "placement_clean_after_eda.csv"
    ),
    index=False
)


print(
    "\nEDA completed."
)

print(
    "All EDA outputs saved to:"
)

print(
    EDA_FOLDER
)


# ============================================================
# ============================================================
#                FEATURE / TARGET SEPARATION
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("11. FEATURE AND TARGET SEPARATION")
print("=" * 80)


# ------------------------------------------------------------
# Columns that should NOT be used for prediction
# ------------------------------------------------------------

columns_to_drop = [

    column

    for column in [

        "StudentID",

        "PlacementStatus",

        "Salary Package",

        "IsAnomaly",

        "CGPA_Tier"

    ]

    if column in df.columns

]


print(
    "\nColumns removed from model features:"
)

for column in columns_to_drop:

    print(
        "-",
        column
    )


X = df.drop(
    columns=columns_to_drop
)

y = df[TARGET]


print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# 19. IDENTIFY MODEL FEATURE TYPES
# ============================================================

numeric_features = (

    X
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()

)


categorical_features = (

    X
    .select_dtypes(

        include=[
            "object",
            "category",
            "string"
        ]

    )
    .columns
    .tolist()

)


print(
    "\nNumerical model features:"
)

for column in numeric_features:

    print(
        "-",
        column
    )


print(
    "\nCategorical model features:"
)

for column in categorical_features:

    print(
        "-",
        column
    )


# ============================================================
# ============================================================
#                     TRAIN TEST SPLIT
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("12. TRAIN-TEST SPLIT")
print("=" * 80)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "X_train:",
    X_train.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test :",
    y_test.shape
)


# ============================================================
# 20. TARGET DISTRIBUTION CHECK
# ============================================================

print(
    "\nTraining target distribution:"
)

print(
    y_train.value_counts(
        normalize=True
    )
)

print(
    "\nTesting target distribution:"
)

print(
    y_test.value_counts(
        normalize=True
    )
)


# ============================================================
# IMPORTANT PREPROCESSING CONCEPT
# ============================================================

print(
    """
============================================================

IMPORTANT:

Train data
    |
    +--> FIT imputer
    |
    +--> FIT encoder
|
    +--> FIT scaler
    |
    +--> Train model


Test data
    |
    +--> TRANSFORM using
         training-fitted objects

Test data must NOT be used to FIT preprocessing.

============================================================
"""
)


# ============================================================
# ============================================================
#                 STANDARD SCALER PIPELINE
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("13. STANDARD SCALER")
print("=" * 80)


# ------------------------------------------------------------
# Numerical pipeline
#
# Missing numerical values
#          ↓
# Median imputation
#          ↓
# StandardScaler
#
# StandardScaler:
#
# z = (x - mean) / standard deviation
# ------------------------------------------------------------

numeric_pipeline_standard = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (
            "scaler",

            StandardScaler()
        )

    ]

)


# ------------------------------------------------------------
# Categorical pipeline
#
# Missing categorical values
#          ↓
# Most frequent imputation
#          ↓
# One Hot Encoding
# ------------------------------------------------------------

try:

    one_hot_encoder = OneHotEncoder(

        handle_unknown="ignore",

        sparse_output=False

    )

except TypeError:

    one_hot_encoder = OneHotEncoder(

        handle_unknown="ignore",

        sparse=False

    )


categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )

        ),

        (
            "encoder",

            one_hot_encoder
        )

    ]

)


# ============================================================
# 21. COLUMN TRANSFORMER
# ============================================================

preprocessor_standard = ColumnTransformer(

    transformers=[

        (
            "numeric",

            numeric_pipeline_standard,

            numeric_features

        ),

        (
            "categorical",

            categorical_pipeline,

            categorical_features

        )

    ],

    remainder="drop"

)


# ============================================================
# 22. FIT ON TRAIN
# ============================================================

print(
    "\nFitting preprocessing on training data..."
)

X_train_standard = (

    preprocessor_standard
    .fit_transform(
        X_train
    )

)


print(
    "Transforming testing data..."
)

X_test_standard = (

    preprocessor_standard
    .transform(
        X_test
    )

)


print(
    "\nProcessed X_train:",
    X_train_standard.shape
)

print(
    "Processed X_test :",
    X_test_standard.shape
)


# ============================================================
# 23. FEATURE NAMES
# ============================================================

feature_names = (

    preprocessor_standard
    .get_feature_names_out()

)


print(
    "\nTotal features after encoding:",
    len(feature_names)
)


print(
    "\nFirst 30 processed feature names:"
)

for feature in feature_names[:30]:

    print(
        feature
    )


# ============================================================
# 24. CONVERT TO DATAFRAME
# ============================================================

X_train_standard_df = pd.DataFrame(

    X_train_standard,

    columns=feature_names,

    index=X_train.index

)


X_test_standard_df = pd.DataFrame(

    X_test_standard,

    columns=feature_names,

    index=X_test.index

)


# ============================================================
# 25. VALIDATE MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("14. PREPROCESSING VALIDATION")
print("=" * 80)


print(
    "Missing values in X_train:",
    X_train_standard_df
    .isnull()
.sum()
    .sum()
)


print(
    "Missing values in X_test:",
    X_test_standard_df
    .isnull()
    .sum()
    .sum()
)


print(
    "Infinite values in X_train:",
    np.isinf(
        X_train_standard_df
    ).sum().sum()
)


print(
    "Infinite values in X_test:",
    np.isinf(
        X_test_standard_df
    ).sum().sum()
)


# ============================================================
# ============================================================
#                     MINMAX SCALER
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("15. MINMAX SCALER")
print("=" * 80)


numeric_pipeline_minmax = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )

        ),

        (
            "scaler",

            MinMaxScaler()
        )

    ]

)


try:

    one_hot_encoder_minmax = OneHotEncoder(

        handle_unknown="ignore",

        sparse_output=False

    )

except TypeError:

    one_hot_encoder_minmax = OneHotEncoder(

        handle_unknown="ignore",

        sparse=False

    )


categorical_pipeline_minmax = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )

        ),

        (
            "encoder",

            one_hot_encoder_minmax
        )

    ]

)


preprocessor_minmax = ColumnTransformer(

    transformers=[

        (
            "numeric",

            numeric_pipeline_minmax,

            numeric_features

        ),

        (
            "categorical",

            categorical_pipeline_minmax,

            categorical_features

        )

    ],

    remainder="drop"

)


# ------------------------------------------------------------
# FIT MinMax on TRAIN only
# ------------------------------------------------------------

X_train_minmax = (

    preprocessor_minmax
    .fit_transform(
        X_train
    )

)


X_test_minmax = (

    preprocessor_minmax
    .transform(
        X_test
    )

)


feature_names_minmax = (

    preprocessor_minmax
    .get_feature_names_out()

)


X_train_minmax_df = pd.DataFrame(

    X_train_minmax,

    columns=feature_names_minmax,

    index=X_train.index

)


X_test_minmax_df = pd.DataFrame(

    X_test_minmax,

    columns=feature_names_minmax,

    index=X_test.index

)


print(
    "MinMax X_train:",
    X_train_minmax_df.shape
)

print(
    "MinMax X_test :",
    X_test_minmax_df.shape
)


# ============================================================
# 26. MINMAX VALIDATION
# ============================================================

numeric_minmax_columns = [

    column

    for column in X_train_minmax_df.columns

    if column.startswith(
        "numeric__"
    )

]


if len(numeric_minmax_columns) > 0:

    print(
        "\nMinMax numerical range:"
    )

    print(

        X_train_minmax_df[
            numeric_minmax_columns
        ]
        .agg(
            ["min", "max"]
        )
        .T

    )


# ============================================================
# ============================================================
#                    SAVE MODEL DATA
# ============================================================
# ============================================================

print("\n" + "=" * 80)
print("16. SAVING MODEL-READY DATA")
print("=" * 80)


# ------------------------------------------------------------
# StandardScaler version
# ------------------------------------------------------------

X_train_standard_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "X_train_standard.csv"

    ),

    index=False

)


X_test_standard_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "X_test_standard.csv"

    ),

    index=False

)


# ------------------------------------------------------------
# MinMax version
# ------------------------------------------------------------

X_train_minmax_df.to_csv(

    os.path.join(
OUTPUT_FOLDER,

        "X_train_minmax.csv"

    ),

    index=False

)


X_test_minmax_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "X_test_minmax.csv"

    ),

    index=False

)


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

y_train.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "y_train.csv"

    ),

    index=False

)


y_test.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "y_test.csv"

    ),

    index=False

)


# ============================================================
# 27. SAVE PREPROCESSORS
# ============================================================

joblib.dump(

    preprocessor_standard,

    os.path.join(

        OUTPUT_FOLDER,

        "standard_preprocessor.pkl"

    )

)


joblib.dump(

    preprocessor_minmax,

    os.path.join(

        OUTPUT_FOLDER,

        "minmax_preprocessor.pkl"

    )

)


# ============================================================
# 28. SAVE FEATURE NAMES
# ============================================================

pd.DataFrame({

    "Feature_Name":
        feature_names

}).to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "processed_feature_names.csv"

    ),

    index=False

)

# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PROJECT STATUS")
print("=" * 80)

print("\nEDA COMPLETED")
print("-" * 60)

print("✓ Dataset inspection")
print("✓ Data types")
print("✓ Statistical summary")
print("✓ Missing value analysis")
print("✓ Duplicate analysis")
print("✓ Target analysis")

print("\nUNIVARIATE ANALYSIS")
print("-" * 60)

print("✓ Important numerical distributions")
print("✓ Important numerical boxplots")
print("✓ Categorical frequency analysis")

print("\nBIVARIATE ANALYSIS")
print("-" * 60)

print("✓ Numerical vs target")
print("✓ Categorical vs target")
print("✓ Numerical vs numerical")

print("\nMULTIVARIATE ANALYSIS")
print("-" * 60)

print("✓ Correlation heatmap")
print("✓ Selected-feature pairplot")

print("\nOUTLIER ANALYSIS")
print("-" * 60)

print("✓ IQR analysis")

print("\nPREPROCESSING")
print("-" * 60)

print("✓ Train-test split")
print("✓ Missing value imputation")
print("✓ One-hot encoding")
print("✓ StandardScaler")
print("✓ MinMaxScaler")
print("✓ Preprocessing validation")

print("\nEDA plots are saved in:")
print(EDA_FOLDER)

print("\nProcessed data is saved in:")
print(OUTPUT_FOLDER)

print("\n" + "=" * 80)
print("MODEL-READY DATA")
print("=" * 80)

print("\nTraining data:")
print(X_train_standard_df.shape)

print("\nTesting data:")
print(X_test_standard_df.shape)

print("\nNumber of processed features:")
print(len(feature_names))

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print("""
MODEL BUILDING

You are now ready to proceed with:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine
6. Gradient Boosting
7. Model Evaluation
8. Confusion Matrix
9. Accuracy
10. Precision
11. Recall
12. F1 Score
13. ROC-AUC
14. Model Comparison
15. Hyperparameter Tuning
""")

print("=" * 80)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 80)
