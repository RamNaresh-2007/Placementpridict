# Principal Component Analysis (PCA) - Dimensionality Reduction & Variance Exploration

## Overview
This folder contains the complete Principal Component Analysis (PCA) pipeline and evaluation assets for student placement prediction data. In earlier sessions, PCA was primarily used as a downstream visualization tool to project high-dimensional clusters onto a 2D plane. This session elevates PCA to the subject itself:
1. Understanding how much global variance is captured across components.
2. Assessing how well 2D and low-dimensional projections preserve real variance structure.
3. Interpreting the principal axes (PC1 & PC2 loadings) to uncover latent latent factors driving student variance.

---

## Directory Structure
```
pca/
├── pca_analysis.py                  # Core PCA pipeline, evaluation, and chart generation
├── pca.py                           # Convenient runner script
├── pca_report.txt                   # Summary report with variance and top loadings
├── scree_and_cumulative_variance.png # Scree plot and cumulative variance curve
├── loadings_biplot.png              # 2D feature loadings biplot (PC1 vs PC2)
├── pca_2d_by_placement.png          # 35,000 students projected to 2D colored by PlacementStatus
├── plots/                           # Dedicated subfolder mirroring all high-res generated charts
└── README.md                        # Documentation
```

---

## Key Findings & Quantitative Results

### 1. Variance Explained
- **PC1 (Academic Performance Factor)**: Explains **64.24%** of total variance. Dominated by semester-wise SGPA (Semesters 2–8) and cumulative CGPA.
- **PC2 (Technical & Practical Skills Factor)**: Explains **11.84%** of total variance. Dominated by CodingTestScore, AptitudeTestScore, Projects, Internships, and Certifications.
- **Combined PC1 + PC2**: Captures **76.08%** of total variance. This justifies why 2D projections used in clustering sessions reasonably represent the core geometry of the data.

### 2. Threshold Analysis
- **80% Variance Threshold**: Requires only **3 components** (retains 80.53% of variance, discarding 85% of dimensions).
- **90% Variance Threshold**: Requires **7 components** (retains 90.90% of variance).
- **95% Variance Threshold**: Requires **10 components** (retains 96.02% of variance).

### 3. Feature Loadings Interpretation
- **PC1 (Academic Axis)**:
  - `SGPA_Sem8`: +0.263
  - `SGPA_Sem7`: +0.262
  - `SGPA_Sem6`: +0.260
  - `SGPA_Sem5`: +0.259
  - `SGPA_Sem4`: +0.258
  - `SGPA_Sem3`: +0.257
  - `SGPA_Sem2`: +0.255
  - `CGPA`: +0.254
- **PC2 (Applied Skills vs Early Semesters Axis)**:
  - `CodingTestScore`: +0.271
  - `AptitudeTestScore`: +0.263
  - `Projects`: +0.258
  - `Internships`: +0.246
  - `SGPA_Sem1`: -0.245
  - `AttendancePercent`: +0.239
  - `SGPA_Sem2`: -0.236
  - `Certifications`: +0.234

---

## Execution
To reproduce the analysis and regenerate all plots and reports:

```powershell
# From workspace root:
.\.venv\Scripts\python "Data/Src/04_Models_and_Pipelines/pca/pca_analysis.py"

# Or using the Session 26 pipeline wrapper:
.\.venv\Scripts\python "Data/Src/04_Models_and_Pipelines/26_pca.py"
```
