"""
Principal Component Analysis (PCA) - Project Root Runner
=========================================================
Convenience entry point for PyCharm and command-line execution.
Forwards directly to placement/Data/Src/04_Models_and_Pipelines/pca/pca_analysis.py
"""

import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(root_dir, "placement", "Data", "Src", "04_Models_and_Pipelines")
pca_dir = os.path.join(pipeline_dir, "pca")
src_dir = os.path.join(root_dir, "placement", "Data", "Src")
app_dir = os.path.join(src_dir, "05_Web_Application")

for p in [app_dir, src_dir, pipeline_dir, pca_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import pca_analysis

if __name__ == "__main__":
    pca_analysis.run_pca_analysis()
