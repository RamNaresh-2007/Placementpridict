"""
Data Leakage Prevention & Robust Model Evaluation - Project Root Runner (Session 28)
====================================================================================
Convenience entry point for PyCharm, VS Code, and command-line execution.
Forwards directly to placement/Data/Src/04_Models_and_Pipelines/data_leakage/data_leakage_pipeline.py
"""

import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(root_dir, "placement", "Data", "Src", "04_Models_and_Pipelines")
leakage_dir = os.path.join(pipeline_dir, "data_leakage")
src_dir = os.path.join(root_dir, "placement", "Data", "Src")
app_dir = os.path.join(src_dir, "05_Web_Application")

for p in [app_dir, src_dir, pipeline_dir, leakage_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import data_leakage_pipeline

if __name__ == "__main__":
    data_leakage_pipeline.run_data_leakage_analysis()
