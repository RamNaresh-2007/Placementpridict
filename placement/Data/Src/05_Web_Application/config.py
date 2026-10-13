"""
PlacementPredict_EDA - Central configuration
==============================================
Every path used by the app is defined once here so nothing gets
hard-coded across routes / scripts.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- data ---
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "processed", "cleaned_data.csv")

# --- static assets served straight to the browser ---
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

# --- generated text reports (same folder src/eda_analysis.py already writes to) ---
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary_report.txt")
LOGISTIC_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "logistic")
HIERARCHICAL_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "hierarchical")
DBSCAN_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "dbscan")
PCA_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca")
PCA_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "pca")
ANOMALY_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection")
ANOMALY_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "anomaly_detection")
DATA_LEAKAGE_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "data_leakage")
LEAKAGE_DIR = DATA_LEAKAGE_DIR
DATA_LEAKAGE_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "data_leakage")
CV_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "cross_validation")
NESTED_CV_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "nested_cv")
THRESHOLD_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "threshold_tuning")
SPLITS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "splits")

RANDOM_STATE = 42

# Full 20 numeric features across academic history and assessment metrics (Sessions 21-25)
NUMERIC_COLUMNS = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications", "AptitudeTestScore",
    "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "ExtraCurricular"
]
ALL_NUMERIC_COLUMNS = NUMERIC_COLUMNS
CORE_NUMERIC_COLUMNS = ["CGPA", "MockInterviewScore", "CodingTestScore", "AptitudeTestScore", "AttendancePercent"]
TARGET_COLUMN = "PlacementStatus"

# --- column groups reused across EDA / preprocessing ---
CATEGORY_COLUMNS = ["Gender", "City", "Stream", "Specialisation"]
YES_NO_COLUMNS = ["Hostel", "HistoryOfBacklogs"]
CATEGORICAL_COLS = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier",
]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
ID_COL = "StudentID"

SECRET_KEY = "dev-key-change-before-deployment"