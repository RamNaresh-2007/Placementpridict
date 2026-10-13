"""
Anomaly & Outlier Detection - Project Root Runner (Session 27)
==============================================================
Convenience entry point for PyCharm and command-line execution.
Forwards directly to placement/Data/Src/04_Models_and_Pipelines/anomaly_detection/anomaly_detection_pipeline.py
"""

import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(root_dir, "placement", "Data", "Src", "04_Models_and_Pipelines")
anomaly_dir = os.path.join(pipeline_dir, "anomaly_detection")
src_dir = os.path.join(root_dir, "placement", "Data", "Src")
app_dir = os.path.join(src_dir, "05_Web_Application")

for p in [app_dir, src_dir, pipeline_dir, anomaly_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import anomaly_detection_pipeline

if __name__ == "__main__":
    anomaly_detection_pipeline.run_anomaly_analysis()
