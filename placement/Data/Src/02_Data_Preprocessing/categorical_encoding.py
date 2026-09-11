
# ==========================================================
# CATEGORICAL ENCODING
# Placement Prediction Dataset
#
# Techniques:
# 1. Categorical Identification
# 2. One-Hot Encoding
# 3. Ordinal Encoding
# 4. Target Encoding
# 5. Accuracy Comparison
# ==========================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# TargetEncoder
try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    TargetEncoder = None


# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

print("\n" + "=" * 75)
print("STEP 1: LOADING DATASET")
print("=" * 75)

# Folder containing this Python file
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Src -> Data
DATA_DIR = os.path.dirname(BASE_DIR)

# ==========================================================
# IMPORTANT:
# Change this filename if your CSV has a different name.
# ==========================================================

CSV_FILENAME = "placement_predict_50k Dataset (2).csv"

file_path = os.path.join(BASE_DIR, "..", "01_Datasets", CSV_FILENAME)

print("\nDataset path:")
print(file_path)

# Check file
if not os.path.exists(file_path):

    print("\nERROR: Dataset file was not found!")
    print("\nPython searched here:")
    print(file_path)

    print("\nFiles available in Data folder:")

    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            print("-", file)

    raise FileNotFoundError(
        "\nPlease change CSV_FILENAME to the correct CSV filename."
    )


# Load dataset
df = pd.read_csv(file_path)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ==========================================================
# STEP 2: DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 75)
print("STEP 2: DATASET INFORMATION")
print("=" * 75)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================================
# STEP 3: IDENTIFY TARGET
# ==========================================================

print("\n" + "=" * 75)
print("STEP 3: IDENTIFY TARGET VARIABLE")
print("=" * 75)

target = "PlacementStatus"

if target not in df.columns:

    print("\nERROR:")
    print("Target column 'PlacementStatus' was not found.")

    print("\nAvailable columns:")
    for column in df.columns:
        print("-", column)

    raise ValueError(
        "Please change the 'target' variable to your actual target column."
    )

print("\nTarget variable:", target)

print("\nTarget categories:")
print(df[target].value_counts())


# ==========================================================
# STEP 4: REMOVE MISSING TARGET VALUES
# ==========================================================

df = df.dropna(subset=[target])


# ==========================================================
# STEP 5: SEPARATE X AND Y
# ==========================================================

X = df.drop(target, axis=1)
X = X.fillna(X.median(numeric_only=True))
y = df[target]

print("\n" + "=" * 75)
print("STEP 5: SEPARATE FEATURES AND TARGET")
print("=" * 75)

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(target)


# ==========================================================
# STEP 6: IDENTIFY CATEGORICAL COLUMNS
# ==========================================================

print("\n" + "=" * 75)
print("STEP 6: IDENTIFY CATEGORICAL COLUMNS")
print("=" * 75)

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:")

if len(categorical_cols) == 0:
    print("No categorical columns found.")

else:
    for column in categorical_cols:
        print("-", column)

print("\nNumber of categorical columns:")
print(len(categorical_cols))

print("\nNumerical columns:")

for column in numerical_cols:
    print("-", column)

print("\nNumber of numerical columns:")
print(len(numerical_cols))


# ==========================================================
# STEP 7: DISPLAY CATEGORIES
# ==========================================================

print("\n" + "=" * 75)
print("STEP 7: CATEGORICAL VALUES")
print("=" * 75)

for column in categorical_cols:

    print("\n" + "-" * 65)

    print("Column:", column)

    print(
        "Number of unique categories:",
        X[column].nunique()
    )

    print("\nCategories:")

    categories = X[column].dropna().unique()

    for category in categories:
        print("-", category)

    print("\nCategory counts:")

    print(
        X[column].value_counts()
    )


# ==========================================================
# STEP 8: TRAIN-TEST SPLIT
# ==========================================================

print("\n" + "=" * 75)
print("STEP 8: TRAIN-TEST SPLIT")
print("=" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

print("\nTraining percentage: 80%")
print("Testing percentage : 20%")

print("\nImportant:")
print("Encoders will be fitted ONLY on training data.")


# ==========================================================
# PART A: ONE-HOT ENCODING
# ==========================================================

print("\n" + "=" * 75)
print("PART A: ONE-HOT ENCODING")
print("=" * 75)

print("""
One-Hot Encoding is normally used for nominal categories.

Example:

Gender:
Male
Female

Encoded as:

Gender_Male
Gender_Female

Male   -> 1 0
Female -> 0 1
""")


# ----------------------------------------------------------
# Create One-Hot Encoder
# ----------------------------------------------------------

try:

    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    # For older scikit-learn versions
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


# ----------------------------------------------------------
# One-Hot Preprocessor
# ----------------------------------------------------------

onehot_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            onehot_encoder,
            categorical_cols
        )
    ],
    remainder="passthrough"
)


# ----------------------------------------------------------
# One-Hot Pipeline
# ----------------------------------------------------------

onehot_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            onehot_preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ----------------------------------------------------------
# Train One-Hot Model
# ----------------------------------------------------------

print("\nTraining One-Hot Encoding model...")

onehot_pipeline.fit(
    X_train,
    y_train
)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred_onehot = onehot_pipeline.predict(
    X_test
)


# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

onehot_accuracy = accuracy_score(
    y_test,
    y_pred_onehot
)

print("\nOne-Hot Encoding Accuracy:")
print(
    round(onehot_accuracy, 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_onehot
    )
)


# ==========================================================
# PART B: ORDINAL ENCODING
# ==========================================================

print("\n" + "=" * 75)
print("PART B: ORDINAL ENCODING")
print("=" * 75)

print("""
Ordinal Encoding assigns numbers to categories.

Example:

Low    -> 0
Medium -> 1
High   -> 2

This method should be used when the categories
have a meaningful order.
""")


# ----------------------------------------------------------
# Create Ordinal Encoder
# ----------------------------------------------------------

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)


# ----------------------------------------------------------
# Ordinal Preprocessor
# ----------------------------------------------------------

ordinal_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            ordinal_encoder,
            categorical_cols
        )
    ],
    remainder="passthrough"
)


# ----------------------------------------------------------
# Ordinal Pipeline
# ----------------------------------------------------------

ordinal_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            ordinal_preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ----------------------------------------------------------
# Train
# ----------------------------------------------------------

print("\nTraining Ordinal Encoding model...")

ordinal_pipeline.fit(
    X_train,
    y_train
)


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred_ordinal = ordinal_pipeline.predict(
    X_test
)


# ----------------------------------------------------------
# Accuracy
# ----------------------------------------------------------

ordinal_accuracy = accuracy_score(
    y_test,
    y_pred_ordinal
)

print("\nOrdinal Encoding Accuracy:")
print(
    round(ordinal_accuracy, 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_ordinal
    )
)


# ==========================================================
# PART C: TARGET ENCODING
# ==========================================================

print("\n" + "=" * 75)
print("PART C: TARGET ENCODING")
print("=" * 75)

if TargetEncoder is None:

    print("""
TargetEncoder is not available in your
current scikit-learn version.

Install/update scikit-learn using:

pip install --upgrade scikit-learn

Then run the program again.
""")

else:

    print("""
Target Encoding replaces each category with
a value based on the target variable.

The encoder is fitted only on training data
to prevent data leakage.
""")

    # ------------------------------------------------------
    # Target Encoder
    # ------------------------------------------------------

    target_encoder = TargetEncoder(
        target_type="auto",
        smooth="auto",
        random_state=42
    )


    # ------------------------------------------------------
    # Target Preprocessor
    # ------------------------------------------------------

    target_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                target_encoder,
                categorical_cols
            )
        ],
        remainder="passthrough"
    )


    # ------------------------------------------------------
    # Target Pipeline
    # ------------------------------------------------------

    target_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                target_preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )


    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    print("\nTraining Target Encoding model...")

    target_pipeline.fit(
        X_train,
        y_train
    )


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    y_pred_target = target_pipeline.predict(
        X_test
    )


    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    target_accuracy = accuracy_score(
        y_test,
        y_pred_target
    )

    print("\nTarget Encoding Accuracy:")
    print(
        round(target_accuracy, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred_target
        )
    )


# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n" + "=" * 75)
print("FINAL ENCODING COMPARISON")
print("=" * 75)

print(
    "\nOne-Hot Encoding Accuracy :",
    round(onehot_accuracy, 4)
)

print(
    "Ordinal Encoding Accuracy :",
    round(ordinal_accuracy, 4)
)

if TargetEncoder is not None:

    print(
        "Target Encoding Accuracy  :",
        round(target_accuracy, 4)
    )


# ==========================================================
# SAVE ENCODED DATASET
# ==========================================================

print("\n" + "=" * 75)
print("SAVING ENCODED DATA")
print("=" * 75)

# Transform the complete dataset using the One-Hot encoder
encoded_features = onehot_preprocessor.fit_transform(
    X
)

# Get encoded column names
try:

    encoded_column_names = (
        onehot_preprocessor
        .get_feature_names_out()
    )

except Exception:

    encoded_column_names = [
        f"Feature_{i}"
        for i in range(
            encoded_features.shape[1]
        )
    ]


# Convert to DataFrame
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoded_column_names
)

# Add target column
encoded_df[target] = y.values

# Output file
output_path = os.path.join(
    DATA_DIR,
    "placement_encoded.csv"
)

encoded_df.to_csv(
    output_path,
    index=False
)

print("\nEncoded dataset saved successfully!")

print("File:")
print(output_path)


# ==========================================================
# COMPLETED
# ==========================================================

print("\n" + "=" * 75)
print("CATEGORICAL ENCODING COMPLETED SUCCESSFULLY")
print("=" * 75)

