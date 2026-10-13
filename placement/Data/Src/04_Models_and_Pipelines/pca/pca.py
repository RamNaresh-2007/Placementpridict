# PCA entrypoint alias
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pca_analysis

if __name__ == "__main__":
    pca_analysis.run_pca_analysis()
