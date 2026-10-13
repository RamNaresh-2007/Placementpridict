# Principal Component Analysis (PCA) - Session 26
# ==============================================================================
# Direct module alias forwarding to pca/pca_analysis.py

import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
pca_subfolder = os.path.join(current_dir, "pca")
if pca_subfolder not in sys.path:
    sys.path.insert(0, pca_subfolder)

# Load module from subfolder to prevent self-referential circular import
_spec = importlib.util.spec_from_file_location("pca_inner_module", os.path.join(pca_subfolder, "pca_analysis.py"))
_pca_inner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pca_inner)

run_pca_analysis = _pca_inner.run_pca_analysis

if __name__ == "__main__":
    run_pca_analysis()
