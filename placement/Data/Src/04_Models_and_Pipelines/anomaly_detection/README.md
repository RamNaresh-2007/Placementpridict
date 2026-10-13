# Unsupervised Anomaly & Outlier Detection - Session 27

## Overview
This module contains the complete Anomaly & Outlier Detection pipeline, mathematical benchmark, and visual evaluation suite for the student placement prediction system.

In previous sessions:
- **Session 21 & 24**: K-Means and Hierarchical Clustering partitioned the feature space into homogeneous student cohorts.
- **Session 25**: DBSCAN spatial clustering revealed that isolated noise points possessed an empirical **93.9% precision overlap** with the ground-truth `IsAnomaly` column.
- **Session 26**: Principal Component Analysis proved that 2 orthogonal components preserve **76.08% of total variance**.

**Session 27 elevates Anomaly Detection to the primary subject**:
1. Comparing 5 distinct algorithmic paradigms: Tree-Partitioning (Isolation Forest), Covariance Ellipsoids (Elliptic Envelope / Mahalanobis Distance), Kernel Hyperplanes (One-Class SVM), Local Reachability Density (Local Outlier Factor), and Density Connectivity (DBSCAN).
2. Quantitatively evaluating detection performance against ground-truth anomalous applicants.
3. Uncovering multidimensional anomaly archetypes (identifying why anomalous students stand out).
4. Providing real-time scoring and diagnostic telemetry to both the Flask Web Suite and Streamlit Dashboard.

---

## Directory Structure
```
anomaly_detection/
├── anomaly_detection_pipeline.py       # Core pipeline, benchmarking, and high-res chart generation
├── anomaly_detection.py                # Convenient local runner script
├── anomaly_report.txt                  # Full textual analytical summary report
├── anomaly_summary.json                # Structured JSON payload for web API consumption
├── 01_anomaly_score_distribution.png   # Isolation Forest decision score KDE and cutoff boundary
├── 02_pca_2d_anomaly_landscape.png     # 2D PCA embedding separating Inliers, True Positives, and False Positives
├── 03_algorithm_comparison_benchmark.png # Multi-metric bar chart comparing 5 algorithms
├── 04_anomaly_feature_profiles.png     # Standardized z-score shifts showing archetype characteristics
├── 05_contamination_sensitivity.png    # Precision, Recall, and F1 dynamics across contamination rates
├── plots/                              # Mirror folder holding all generated high-res plots
└── README.md                           # Technical documentation and mathematical guide
```

---

## Benchmark Results & Empirical Findings

| Model | Algorithmic Paradigm | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Scalability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Elliptic Envelope** | Robust Covariance (FastMCD) | **0.9884** | **0.9447** | **80.6%** | **91.6%** | **0.8571** | Medium ($O(n \cdot p^2)$) |
| **DBSCAN Noise** | Density Reachability | N/A | N/A | **93.9%** | **69.5%** | **0.7985** | $O(n^2)$ distance matrix |
| **Isolation Forest** | Random Recursive Tree Cuts | **0.9305** | **0.3882** | **35.1%** | **35.6%** | **0.3535** | High ($O(n \log n)$) |
| **One-Class SVM** | Non-Linear RBF Boundary | **0.9468** | **0.4532** | **39.4%** | **46.1%** | **0.4251** | Medium ($O(n^2)$) |
| **Local Outlier Factor** | K-Nearest Neighbor Density | **0.8490** | **0.2970** | **27.4%** | **31.2%** | **0.2918** | Medium ($O(n^2)$) |

### Key Mathematical Insights:
1. **Covariance Contradiction (Why Elliptic Envelope Wins)**:
   The ground truth anomalies in this dataset are characterized by multidimensional feature contradictions rather than isolated single-feature univariate outliers. For instance, anomalous students frequently exhibit near-perfect academic credentials (CGPA ~9.7, 5 projects, 3 internships) while simultaneously suffering critical attendance deficits (~54% attendance) or zero coding scores. Robust covariance methods detect this inter-variable breakdown immediately.
2. **Precision vs. Recall Tradeoff**:
   - For **zero false-alarm deployment**, DBSCAN noise is optimal (93.9% precision).
   - For **comprehensive anomaly audits**, Elliptic Envelope captures 91.6% of all true anomalies.
   - For **real-time streaming**, Isolation Forest evaluates incoming students in sub-millisecond latency.

---

## Execution
To run the anomaly detection pipeline, generate all visual artifacts, and update API assets:

```powershell
# From project root:
.\.venv\Scripts\python "anomaly_detection.py"

# Or using the Session 27 pipeline wrapper:
.\.venv\Scripts\python "placement/Data/Src/04_Models_and_Pipelines/27_anomaly_detection.py"

# Or running the pipeline directly:
.\.venv\Scripts\python "placement/Data/Src/04_Models_and_Pipelines/anomaly_detection/anomaly_detection_pipeline.py"
```
