# Session 26 - Principal Component Analysis (PCA)
# ==================================================================================
# Wrapper delegating to pca/pca_analysis.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
pca_dir = os.path.join(current_dir, "pca")
if pca_dir not in sys.path:
    sys.path.insert(0, pca_dir)

import pca_analysis

if __name__ == "__main__":
    pca_analysis.run_pca_analysis()
