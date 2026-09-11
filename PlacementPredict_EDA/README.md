# PlacementPredict - EDA Module

Exploratory Data Analysis on the 50,000-row student placement dataset.

## Folder structure

```
PlacementPredict_EDA/
├── data/
│   └── placement_data.csv         <- dataset (50,000 rows, 32 columns)
├── src/
│   └── eda_analysis.py            <- run this file
├── outputs/
│   ├── plots/                     <- 11 PNG charts generated on each run
│   └── reports/
│       └── eda_summary_report.txt <- full text summary of every step
├── requirements.txt
└── README.md
```

## How to run (PyCharm / Windows)

1. Open the `PlacementPredict_EDA` folder as a PyCharm project.
2. Open a terminal inside PyCharm and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run `src/eda_analysis.py` (right-click → Run, or `python src/eda_analysis.py`).
4. Check `outputs/plots/` for charts and `outputs/reports/eda_summary_report.txt`
   for the full printed analysis.

No Jupyter/browser environment needed — it's a plain `.py` script.

## What the script does

| # | Step | What it checks |
|---|------|-----------------|
| 1 | Load data | Reads the CSV, reports shape |
| 2 | First look | Head rows, dtypes |
| 3 | Missing values | Count + % per column, bar chart |
| 4 | Duplicates | Full-row duplicates, duplicate StudentIDs |
| 5 | Descriptive stats | Mean/std/min/max/skew for numeric columns |
| 6 | Categorical frequencies | Value counts for every categorical column |
| 7 | Target balance | PlacementStatus 0 vs 1 counts and chart |
| 8 | Distributions | Histograms of CGPA, Attendance, test scores |
| 9 | Outliers | IQR-based outlier counts + boxplots |
| 10 | Correlation | Heatmap + ranked correlation with PlacementStatus |
| 11 | Categorical vs target | Placement rate % by CollegeTier, Stream, etc. |
| 12 | Numeric vs target | Boxplots of CGPA/Attendance/CodingScore by placement |
| 13 | Pairwise relationships | Sampled pairplot of key numeric features |
| 14 | Salary analysis | Salary distribution and by-college-tier boxplot (placed only) |
| 15 | Anomaly flag | IsAnomaly counts and rate |
| 16 | SGPA trend | Semester-wise average SGPA, placed vs not placed |

## Key columns in the dataset

- **Demographics**: Gender, City, CollegeTier, Stream, Specialisation, Hostel
- **Academics**: SGPA_Sem1–8, CGPA, CGPA_Tier, AttendancePercent, HistoryOfBacklogs
- **Activity**: Internships, Projects, Workshops, Certifications, Publications, ExtraCurricular
- **Assessment scores**: AptitudeTestScore, SoftSkillsRating, CodingTestScore, MockInterviewScore
- **Targets**: PlacementStatus (0/1), Salary Package, IsAnomaly

## Notes on missing data

`Workshops`, `AptitudeTestScore`, `SoftSkillsRating`, and `MockInterviewScore`
contain missing values (roughly 6-10% each). This EDA reports them but does not
impute/clean them — that belongs in the preprocessing module (Module 2).
