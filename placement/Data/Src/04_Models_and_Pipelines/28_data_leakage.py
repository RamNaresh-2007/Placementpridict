# Session 28 - Data Leakage Prevention & Robust Model Evaluation
# ==================================================================================
# Wrapper delegating to data_leakage/data_leakage_pipeline.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
leakage_dir = os.path.join(current_dir, "data_leakage")
if leakage_dir not in sys.path:
    sys.path.insert(0, leakage_dir)

import data_leakage_pipeline

if __name__ == "__main__":
    data_leakage_pipeline.run_data_leakage_analysis()
