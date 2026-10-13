import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import anomaly_detection_pipeline

if __name__ == "__main__":
    anomaly_detection_pipeline.run_anomaly_analysis()
