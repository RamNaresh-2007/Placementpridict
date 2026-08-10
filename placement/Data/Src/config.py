"""
PlacementPredict_EDA - Central configuration
==============================================
Every path used by the app is defined once here so nothing gets
hard-coded across routes / scripts.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- data ---
RAW_DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset (2).csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "processed", "cleaned_data.csv")

# --- static assets served straight to the browser ---
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

# --- generated text reports (same folder src/eda_analysis.py already writes to) ---
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary_report.txt")

# --- column groups reused across EDA / preprocessing ---
CATEGORICAL_COLS = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier",
]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
ID_COL = "StudentID"

SECRET_KEY = "dev-key-change-before-deployment"