# Session 27 - Unsupervised Anomaly & Outlier Detection
# ==================================================================================
# Wrapper delegating to anomaly_detection/anomaly_detection_pipeline.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
anomaly_dir = os.path.join(current_dir, "anomaly_detection")
if anomaly_dir not in sys.path:
    sys.path.insert(0, anomaly_dir)

import anomaly_detection_pipeline

if __name__ == "__main__":
    anomaly_detection_pipeline.run_anomaly_analysis()
