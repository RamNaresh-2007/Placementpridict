import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from urllib.parse import unquote

app = Flask(__name__)

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")

_CACHED_DF = None

def get_data():
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF.copy()
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    
    if "PlacementStatus" in df.columns:
        df["PlacementStatus"] = df["PlacementStatus"].map({1: "Placed", 0: "Not Placed"}).fillna(df["PlacementStatus"])
        
    _CACHED_DF = df
    return _CACHED_DF.copy()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    try:
        df = get_data()
        total_students = len(df)

        # Calculate Missing Values and Cleaned Data
        total_missing = int(df.isnull().sum().sum())
        cleaned_records = len(df.dropna())

        # Placement Rate
        placed_count = len(df[df["PlacementStatus"] == "Placed"])
        placement_rate = round((placed_count / total_students) * 100, 1) if total_students > 0 else 0

        # Avg CGPA
        avg_cgpa = round(df["CGPA"].mean(), 2) if "CGPA" in df.columns else 0

        # Avg Salary Package
        avg_salary = "N/A"
        if "Salary Package" in df.columns:
            salary_numeric = pd.to_numeric(df["Salary Package"], errors='coerce')
            if not pd.isna(salary_numeric.mean()):
                # Exclude 0.00 (unplaced) for a more accurate average of placed students
                placed_salary = salary_numeric[salary_numeric > 0]
                if not placed_salary.empty:
                    avg_salary = f"{placed_salary.mean():.2f} LPA"
                else:
                    avg_salary = f"{salary_numeric.mean():.2f} LPA"

        # CGPA Histogram Data
        cgpa_hist = {}
        if "CGPA" in df.columns:
            counts, bins = np.histogram(df["CGPA"].dropna(), bins=10)
            cgpa_hist = {
                "labels": [f"{round(bins[i], 1)}-{round(bins[i+1], 1)}" for i in range(len(counts))],
                "data": counts.tolist()
            }

        # Placement Status Data
        placement_status_data = {}
        if "PlacementStatus" in df.columns:
            counts = df["PlacementStatus"].value_counts()
            placement_status_data = {
                "labels": counts.index.tolist(),
                "data": counts.values.tolist()
            }

        # Stream Data
        stream_data = {}
        if "Stream" in df.columns:
            counts = df["Stream"].value_counts()
            stream_data = {
                "labels": counts.index.tolist(),
                "data": counts.values.tolist()
            }

        return jsonify({
            "kpis": {
                "total_students": total_students,
                "placement_rate": placement_rate,
                "avg_cgpa": avg_cgpa,
                "avg_salary": avg_salary,
                "missing_values": total_missing,
                "cleaned_records": cleaned_records
            },
            "charts": {
                "cgpa_hist": cgpa_hist,
                "placement_status": placement_status_data,
                "stream_data": stream_data
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/students")
def api_students():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 50))
        filter_type = request.args.get("filter", "all")

        df = get_data()
        
        # Apply filter
        if filter_type == "cleaned":
            df = df.dropna()
        elif filter_type == "missing":
            df = df[df.isnull().any(axis=1)]

        total = len(df)

        # Calculate pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        # Slice dataframe and replace NaNs with None for JSON serialization
        df_slice = df.iloc[start_idx:end_idx].replace({np.nan: None})
        data = df_slice.to_dict(orient="records")

        return jsonify({
            "data": data,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Output Directory Paths
EDA_DIR = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "plots")
EDA_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "EDA_Outputs")
PIPELINE_EDA_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "EDA_Outputs")
DBSCAN_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "dbscan")
HIER_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "hierarchical")
PCA_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "pca")
PCA_ALT_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca")
PCA_STATIC_DIR = os.path.join(BASE_DIR, "static", "plots", "pca")
ANOMALY_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "anomaly_detection")
ANOMALY_ALT_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection")
ANOMALY_STATIC_DIR = os.path.join(BASE_DIR, "static", "plots", "anomaly_detection")
LOGISTIC_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "logistic")
DATA_LEAKAGE_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "data_leakage")
DATA_LEAKAGE_ALT_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "data_leakage")
DATA_LEAKAGE_STATIC_DIR = os.path.join(BASE_DIR, "static", "plots", "data_leakage")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Classification")
REGRESSION_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Regression")

@app.route("/api/plots")
def api_plots():
    try:
        plots_meta = []
        seen_filenames = set()

        def add_plot(filename, title, category, category_slug, description):
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                plots_meta.append({
                    "filename": filename,
                    "title": title,
                    "category": category,
                    "category_slug": category_slug,
                    "description": description
                })

        # 1. DBSCAN plots
        if os.path.exists(DBSCAN_DIR):
            for f in sorted(os.listdir(DBSCAN_DIR)):
                if f.endswith('.png'):
                    title = "DBSCAN: K-Distance Elbow Plot (eps=2.88)" if "k_distance" in f else "DBSCAN: 2D PCA Clusters vs Outlier Noise"
                    add_plot(
                        f"dbscan/{f}",
                        title,
                        "DBSCAN & Clustering",
                        "dbscan",
                        "Density-based spatial clustering isolating dense core student regions from sparse noise points."
                    )

        # 2. Hierarchical plots
        if os.path.exists(HIER_DIR):
            for f in sorted(os.listdir(HIER_DIR)):
                if f.endswith('.png'):
                    title = f"[Hierarchical] {f.replace('.png', '').replace('_', ' ').title()}"
                    add_plot(
                        f"hierarchical/{f}",
                        title,
                        "DBSCAN & Clustering",
                        "dbscan",
                        "Agglomerative hierarchical clustering dendrograms, silhouette score cuts, and cluster profiles."
                    )

        # 3. PCA plots
        for pdir in [PCA_DIR, PCA_ALT_DIR, PCA_STATIC_DIR]:
            if os.path.exists(pdir):
                for f in sorted(os.listdir(pdir)):
                    if f.endswith('.png'):
                        title = f"[PCA] {f.replace('.png', '').replace('_', ' ').title()}"
                        add_plot(
                            f"pca/{f}",
                            title,
                            "PCA Projections",
                            "pca",
                            "Principal Component Analysis variance scree plot, loadings biplot, and 2D student space."
                        )

        # 3b. Anomaly & Outlier Intelligence Plots
        for adir in [ANOMALY_DIR, ANOMALY_ALT_DIR, ANOMALY_STATIC_DIR]:
            if os.path.exists(adir):
                for f in sorted(os.listdir(adir)):
                    if f.endswith('.png'):
                        title = f"[Anomaly] {f.replace('.png', '').replace('_', ' ').title()}"
                        add_plot(
                            f"anomaly_detection/{f}",
                            title,
                            "Anomaly & Outlier Intelligence",
                            "anomaly",
                            "Unsupervised anomaly detection score distributions, 2D PCA anomaly landscape, benchmark comparisons, and feature deviation profiles."
                        )
                break

        # 3c. Data Leakage & Robust Cross-Validation Plots (Session 28)
        for ldir in [DATA_LEAKAGE_DIR, DATA_LEAKAGE_ALT_DIR, DATA_LEAKAGE_STATIC_DIR]:
            if os.path.exists(ldir):
                for f in sorted(os.listdir(ldir)):
                    if f.endswith('.png'):
                        title = f"[Data Leakage] {f.replace('.png', '').replace('_', ' ').title()}"
                        add_plot(
                            f"data_leakage/{f}",
                            title,
                            "Data Leakage & Cross-Validation",
                            "leakage",
                            "Vault-lock holdout isolation, nested cross-validation (5x3), and empirical proof of distribution contamination."
                        )
                break

        # 4. Supervised Classification Confusion Matrices
        if os.path.exists(CLASSIFICATION_DIR):
            for f in sorted(os.listdir(CLASSIFICATION_DIR)):
                if f.endswith('_confusion_matrix.png'):
                    title = f"{f.replace('_confusion_matrix.png', '').replace('_', ' ').title()} - Confusion Matrix"
                    add_plot(
                        f"classification/{f}",
                        title,
                        "Confusion Matrices",
                        "classification",
                        "Supervised classification confusion matrix evaluating precision, recall, and accuracy."
                    )

        # 5. Salary Regression Diagnostic plots
        if os.path.exists(REGRESSION_DIR):
            for f in sorted(os.listdir(REGRESSION_DIR)):
                if f.endswith('.png'):
                    title = f"[Regression] {f.replace('.png', '').replace('_', ' ').title()}"
                    add_plot(
                        f"regression/{f}",
                        title,
                        "Salary Regression",
                        "regression",
                        "Linear regression actual vs predicted packages, residual plots, and gradient descent cost curve."
                    )

        # 6. Logistic Regression plots
        if os.path.exists(LOGISTIC_DIR):
            for f in sorted(os.listdir(LOGISTIC_DIR)):
                if f.endswith('.png'):
                    title = f"[Logistic] {f.replace('.png', '').replace('_', ' ').title()}"
                    add_plot(
                        f"logistic/{f}",
                        title,
                        "Confusion Matrices",
                        "classification",
                        "Logistic regression sigmoid probability curves and 2D decision boundary visualizations."
                    )

        # 7. Comprehensive Overview EDA Outputs
        if os.path.exists(EDA_OUTPUTS_DIR):
            for f in sorted(os.listdir(EDA_OUTPUTS_DIR)):
                if f.endswith('.png'):
                    title = f"[EDA Overview] {f.replace('.png', '').replace('_', ' ').title()}"
                    add_plot(
                        f"eda_outputs/{f}",
                        title,
                        "Overview EDA & Heatmaps",
                        "eda_overview",
                        f"Comprehensive institutional distribution, correlation heatmap, and bivariate interaction chart."
                    )

        # 8. Pipeline EDA Outputs
        if os.path.exists(PIPELINE_EDA_DIR):
            for f in sorted(os.listdir(PIPELINE_EDA_DIR)):
                if f.endswith('.png'):
                    title = f"[Pipeline EDA] {f.replace('.png', '').replace('_', ' ').title()}"
                    add_plot(
                        f"pipeline_eda/{f}",
                        title,
                        "Overview EDA & Heatmaps",
                        "eda_overview",
                        f"Data preprocessing pipeline distribution analysis, boxplot, and target correlation."
                    )

        # 9. Individual Feature EDA plots
        if os.path.exists(EDA_DIR):
            for f in sorted(os.listdir(EDA_DIR)):
                if f.endswith('.png'):
                    lower = f.lower()
                    if any(k in lower for k in ["cgpa", "sgpa"]):
                        cat = "Academic (CGPA/SGPA)"
                        cat_slug = "academic"
                    elif any(k in lower for k in ["coding", "aptitude", "mock", "softskill", "workshop", "project", "certification", "publication", "internship"]):
                        cat = "Skills & Technical Tests"
                        cat_slug = "skills"
                    elif any(k in lower for k in ["gender", "tier", "stream", "city", "hostel", "backlog", "specialisation"]):
                        cat = "Demographics & Tiers"
                        cat_slug = "demographics"
                    elif any(k in lower for k in ["anomaly", "placement"]):
                        cat = "Placement Outcomes"
                        cat_slug = "outcomes"
                    else:
                        cat = "General EDA"
                        cat_slug = "other_eda"

                    title = f.replace('.png', '').replace('_', ' ').title()
                    add_plot(
                        f"eda/{f}",
                        title,
                        cat,
                        cat_slug,
                        f"Exploratory distribution and variance analysis for {title}."
                    )

        all_plots_filenames = [p["filename"] for p in plots_meta]

        return jsonify({
            "status": "success",
            "total_plots": len(plots_meta),
            "plots_meta": plots_meta,
            "plots": all_plots_filenames,
            "all_plots": all_plots_filenames
        })
    except Exception as e:
        return jsonify({"error": str(e), "plots": [], "plots_meta": []}), 500

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    filename = unquote(filename)

    prefix_map = [
        ("dbscan/", DBSCAN_DIR),
        ("hierarchical/", HIER_DIR),
        ("pca/", PCA_DIR),
        ("anomaly_detection/", ANOMALY_DIR),
        ("anomaly/", ANOMALY_DIR),
        ("data_leakage/", DATA_LEAKAGE_DIR),
        ("leakage/", DATA_LEAKAGE_DIR),
        ("classification/", CLASSIFICATION_DIR),
        ("regression/", REGRESSION_DIR),
        ("logistic/", LOGISTIC_DIR),
        ("eda_outputs/", EDA_OUTPUTS_DIR),
        ("pipeline_eda/", PIPELINE_EDA_DIR),
        ("eda/", EDA_DIR)
    ]

    for prefix, target_dir in prefix_map:
        if filename.startswith(prefix):
            real_file = filename[len(prefix):]
            if os.path.exists(os.path.join(target_dir, real_file)):
                return send_from_directory(target_dir, real_file)
            if prefix == "pca/":
                for alt in [PCA_ALT_DIR, PCA_STATIC_DIR]:
                    if os.path.exists(os.path.join(alt, real_file)):
                        return send_from_directory(alt, real_file)
            if prefix in ["anomaly_detection/", "anomaly/"]:
                for alt in [ANOMALY_ALT_DIR, ANOMALY_STATIC_DIR]:
                    if os.path.exists(os.path.join(alt, real_file)):
                        return send_from_directory(alt, real_file)
            if prefix in ["data_leakage/", "leakage/"]:
                for alt in [DATA_LEAKAGE_ALT_DIR, DATA_LEAKAGE_STATIC_DIR]:
                    if os.path.exists(os.path.join(alt, real_file)):
                        return send_from_directory(alt, real_file)

    # Fallback lookup across all known plot directories
    all_directories = [
        EDA_DIR,
        EDA_OUTPUTS_DIR,
        PIPELINE_EDA_DIR,
        DBSCAN_DIR,
        HIER_DIR,
        PCA_DIR,
        PCA_ALT_DIR,
        PCA_STATIC_DIR,
        ANOMALY_DIR,
        ANOMALY_ALT_DIR,
        ANOMALY_STATIC_DIR,
        DATA_LEAKAGE_DIR,
        DATA_LEAKAGE_ALT_DIR,
        DATA_LEAKAGE_STATIC_DIR,
        LOGISTIC_DIR,
        CLASSIFICATION_DIR,
        REGRESSION_DIR
    ]

    bare_name = os.path.basename(filename)
    for directory in all_directories:
        if os.path.exists(os.path.join(directory, bare_name)):
            return send_from_directory(directory, bare_name)

    return jsonify({"error": f"Plot '{filename}' not found"}), 404

@app.route("/api/dbscan")
def api_dbscan():
    try:
        report_path = os.path.join(DBSCAN_DIR, "dbscan_report.txt")
        report_text = ""
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_text = f.read()

        return jsonify({
            "status": "success",
            "model": "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)",
            "subsample_size": 5000,
            "min_samples": 40,
            "eps": 2.88,
            "clusters_found": 1,
            "core_points": 4675,
            "core_points_pct": 93.5,
            "noise_points": 114,
            "noise_points_pct": 2.3,
            "anomalies_flagged_both": 107,
            "noise_that_is_anomaly_pct": 93.9,
            "anomaly_detected_pct": 69.5,
            "placement_rate_clustered": 65.3,
            "placement_rate_noise": 71.1,
            "sensitivity": [
                {"eps": 1.44, "clusters": 1, "noise": 4960, "noise_pct": 99.2},
                {"eps": 2.88, "clusters": 1, "noise": 114, "noise_pct": 2.3},
                {"eps": 5.76, "clusters": 1, "noise": 0, "noise_pct": 0.0}
            ],
            "plots": [
                {
                    "id": "k_distance",
                    "title": "K-Distance Elbow Plot (eps = 2.88 Knee)",
                    "url": "/plots/dbscan/k_distance_plot.png",
                    "caption": "Sorted distance to 40th nearest neighbor; kneedle curvature pinpoints optimal epsilon."
                },
                {
                    "id": "clusters_pca",
                    "title": "DBSCAN Clusters vs Outlier Noise in 2D PCA Space",
                    "url": "/plots/dbscan/dbscan_clusters_pca_2d.png",
                    "caption": "Dimensional compression showing 4,675 dense core students vs 114 isolated noise students (93.9% true anomalies)."
                }
            ],
            "report": report_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pca")
def api_pca():
    try:
        # Check summary json first
        summary_paths = [
            os.path.join(PCA_DIR, "pca_summary.json"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca", "pca_summary.json")
        ]
        summary_data = None
        for sp in summary_paths:
            if os.path.exists(sp):
                with open(sp, "r") as f:
                    summary_data = json.load(f)
                break

        # Check report text
        report_paths = [
            os.path.join(PCA_DIR, "pca_report.txt"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca", "pca_report.txt")
        ]
        report_text = ""
        for rp in report_paths:
            if os.path.exists(rp):
                with open(rp, "r") as f:
                    report_text = f.read()
                break

        if summary_data is not None:
            summary_data["report"] = report_text
            return jsonify(summary_data)

        # Fallback values if summary json has not yet been generated
        return jsonify({
            "status": "success",
            "method": "Principal Component Analysis (PCA)",
            "total_students": 35000,
            "total_features": 20,
            "total_components": 20,
            "pc1_variance_pct": 64.24,
            "pc2_variance_pct": 11.84,
            "pc1_pc2_cumulative_pct": 76.08,
            "thresholds": {
                "80%": 3,
                "90%": 7,
                "95%": 10
            },
            "top_pc1_loadings": [
                {"feature": "SGPA_Sem8", "loading": 0.263},
                {"feature": "SGPA_Sem7", "loading": 0.262},
                {"feature": "SGPA_Sem6", "loading": 0.260},
                {"feature": "SGPA_Sem5", "loading": 0.259},
                {"feature": "SGPA_Sem4", "loading": 0.258},
                {"feature": "SGPA_Sem3", "loading": 0.257},
                {"feature": "SGPA_Sem2", "loading": 0.255},
                {"feature": "CGPA", "loading": 0.254}
            ],
            "top_pc2_loadings": [
                {"feature": "CodingTestScore", "loading": 0.271},
                {"feature": "AptitudeTestScore", "loading": 0.263},
                {"feature": "Projects", "loading": 0.258},
                {"feature": "Internships", "loading": 0.246},
                {"feature": "SGPA_Sem1", "loading": -0.245},
                {"feature": "AttendancePercent", "loading": 0.239},
                {"feature": "SGPA_Sem2", "loading": -0.236},
                {"feature": "Certifications", "loading": 0.234}
            ],
            "plots": [
                {
                    "id": "scree_variance",
                    "filename": "pca/scree_and_cumulative_variance.png",
                    "title": "Scree Plot & Cumulative Variance Curve",
                    "caption": "Scree bar chart and cumulative variance progression across components with 80%, 90%, and 95% thresholds."
                },
                {
                    "id": "loadings_biplot",
                    "filename": "pca/loadings_biplot.png",
                    "title": "PCA Loadings Biplot (PC1 vs PC2)",
                    "caption": "Vector representation of 20 original features projected onto principal components PC1 and PC2."
                },
                {
                    "id": "pca_2d_scatter",
                    "filename": "pca/pca_2d_by_placement.png",
                    "title": "2D Student Landscape by Placement Status",
                    "caption": "All 35,000 students projected into 2D principal coordinate space, colored by actual placement outcome."
                }
            ],
            "report": report_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/anomaly")
def api_anomaly():
    try:
        summary_paths = [
            os.path.join(ANOMALY_DIR, "anomaly_summary.json"),
            os.path.join(ANOMALY_ALT_DIR, "anomaly_summary.json"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection", "anomaly_summary.json")
        ]
        summary_data = None
        for sp in summary_paths:
            if os.path.exists(sp):
                with open(sp, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                break

        report_paths = [
            os.path.join(ANOMALY_DIR, "anomaly_report.txt"),
            os.path.join(ANOMALY_ALT_DIR, "anomaly_report.txt"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection", "anomaly_report.txt")
        ]
        report_text = ""
        for rp in report_paths:
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    report_text = f.read()
                break

        if summary_data is not None:
            summary_data["report"] = report_text
            return jsonify(summary_data)

        # Fallback values if summary json has not yet been generated
        return jsonify({
            "status": "success",
            "method": "Unsupervised Anomaly & Outlier Detection Suite",
            "session": "Session 27",
            "total_students": 35000,
            "total_features": 20,
            "ground_truth_anomalies": 1208,
            "ground_truth_pct": 3.45,
            "primary_model": {
                "name": "Isolation Forest",
                "full_cohort_roc_auc": 0.9305,
                "full_cohort_pr_auc": 0.3882,
                "full_cohort_f1": 0.3535,
                "full_cohort_precision": 35.1,
                "full_cohort_recall": 35.6,
                "detected_anomalies": 1225,
                "detected_pct": 3.5
            },
            "best_covariance_model": {
                "name": "Elliptic Envelope (Mahalanobis)",
                "roc_auc": 0.9884,
                "precision_pct": 80.6,
                "recall_pct": 91.6,
                "f1_score": 0.8571
            },
            "best_density_model": {
                "name": "DBSCAN Noise Filter",
                "precision_pct": 93.9,
                "recall_pct": 69.5,
                "f1_score": 0.7985
            },
            "report": report_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/leakage")
def api_leakage():
    try:
        summary_paths = [
            os.path.join(DATA_LEAKAGE_DIR, "leakage_summary.json"),
            os.path.join(DATA_LEAKAGE_ALT_DIR, "leakage_summary.json"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "data_leakage", "leakage_summary.json")
        ]
        summary_data = None
        for sp in summary_paths:
            if os.path.exists(sp):
                with open(sp, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                break

        report_paths = [
            os.path.join(DATA_LEAKAGE_DIR, "leakage_report.txt"),
            os.path.join(DATA_LEAKAGE_ALT_DIR, "leakage_report.txt"),
            os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "data_leakage", "leakage_report.txt")
        ]
        report_text = ""
        for rp in report_paths:
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    report_text = f.read()
                break

        if summary_data is not None:
            summary_data["report"] = report_text
            summary_data["status"] = "success"
            return jsonify(summary_data)

        # Fallback values if summary json is not yet populated
        return jsonify({
            "status": "success",
            "title": "Session 28: Data Leakage Prevention, Vault-Locking & Nested CV",
            "vault_lock": {
                "total_samples": 5000,
                "dev_size": 4000,
                "vault_test_size": 1000,
                "cv_acc_mean": 0.8965,
                "cv_acc_std": 0.0155,
                "cv_f1_mean": 0.9220,
                "cv_f1_std": 0.0119,
                "cv_roc_auc_mean": 0.9507,
                "cv_roc_auc_std": 0.0054,
                "vault_test_accuracy": 0.8950,
                "vault_test_f1": 0.9211,
                "vault_test_roc_auc": 0.9411
            },
            "nested_cv": {
                "nested_f1_mean": 0.9213,
                "nested_f1_std": 0.0058,
                "nested_acc_mean": 0.8944,
                "nested_acc_std": 0.0067,
                "nested_auc_mean": 0.9484,
                "nested_auc_std": 0.0089,
                "non_nested_optimistic_f1": 0.9221,
                "tuning_optimism_delta": 0.0007,
                "fold_details": [
                    {"fold": 1, "best_C": 0.1, "inner_cv_best_f1": 0.9224, "outer_val_f1": 0.9174, "outer_val_acc": 0.891},
                    {"fold": 2, "best_C": 0.1, "inner_cv_best_f1": 0.9249, "outer_val_f1": 0.9132, "outer_val_acc": 0.885},
                    {"fold": 3, "best_C": 0.01, "inner_cv_best_f1": 0.9222, "outer_val_f1": 0.9215, "outer_val_acc": 0.893},
                    {"fold": 4, "best_C": 0.01, "inner_cv_best_f1": 0.9208, "outer_val_f1": 0.9299, "outer_val_acc": 0.905},
                    {"fold": 5, "best_C": 0.01, "inner_cv_best_f1": 0.9228, "outer_val_f1": 0.9248, "outer_val_acc": 0.898}
                ]
            },
            "report": report_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/clustering")
def api_clustering():
    try:
        hier_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "hierarchical")
        report_path = os.path.join(hier_dir, "hierarchical_report.txt")
        report_text = ""
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_text = f.read()
        
        return jsonify({
            "status": "success",
            "optimal_k": 2,
            "silhouette_score": 0.4463,
            "personas": {
                "cluster_1": {
                    "name": "At-Risk Cohort",
                    "student_count": 402,
                    "placement_rate_pct": 5.72,
                    "mean_cgpa": 5.39,
                    "mean_coding": 26.18,
                    "mean_mock": 31.93,
                    "mean_aptitude": 46.18,
                    "mean_attendance": 60.71
                },
                "cluster_2": {
                    "name": "Placement-Ready Cohort",
                    "student_count": 1598,
                    "placement_rate_pct": 82.23,
                    "mean_cgpa": 7.74,
                    "mean_coding": 65.98,
                    "mean_mock": 61.61,
                    "mean_aptitude": 74.45,
                    "mean_attendance": 81.03
                }
            },
            "report": report_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        import joblib
        data = request.get_json() or {}
        
        cgpa = float(data.get("cgpa", 7.5))
        coding = float(data.get("coding_test_score", 65.0))
        aptitude = float(data.get("aptitude_test_score", 70.0))
        mock = float(data.get("mock_interview_score", 65.0))
        attendance = float(data.get("attendance_percent", 85.0))
        internships = int(data.get("internships", 1))
        projects = int(data.get("projects", 2))
        certs = int(data.get("certifications", 1))
        gender = str(data.get("gender", "Male"))
        tier = str(data.get("college_tier", "Tier2"))
        stream = str(data.get("stream", "CS"))
        backlogs = str(data.get("history_of_backlogs", "No"))
        
        model_path = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Classification", "logistic_regression.pkl")
        reg_model_path = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Regression", "linear_regression.pkl")
        reg_prep_path = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Regression", "regression_preprocessor.pkl")
        
        # Soft skills in dataset is on 1.0 to 5.0 scale (median 3.2).
        # We derive it naturally from mock interview score:
        soft_skills = max(1.0, min(5.0, round(mock / 20.0, 1)))

        input_dict = {
            'SGPA_Sem1': [cgpa], 'SGPA_Sem2': [cgpa], 'SGPA_Sem3': [cgpa], 'SGPA_Sem4': [cgpa],
            'SGPA_Sem5': [cgpa], 'SGPA_Sem6': [cgpa], 'SGPA_Sem7': [cgpa], 'SGPA_Sem8': [cgpa],
            'CGPA': [cgpa], 'AttendancePercent': [attendance],
            'Internships': [internships], 'Projects': [projects], 
            'Workshops': [1], 'Certifications': [certs], 'Publications': [0],
            'AptitudeTestScore': [aptitude], 'SoftSkillsRating': [soft_skills],
            'CodingTestScore': [coding], 'MockInterviewScore': [mock], 'ExtraCurricular': [1],
            'Gender': [gender], 'City': ['Pune'], 'CollegeTier': [tier],
            'Stream': [stream], 'Specialisation': ['AI'], 'Hostel': ['No'], 'HistoryOfBacklogs': [backlogs]
        }
        input_df = pd.DataFrame(input_dict)
        
        prob = 0.5
        if os.path.exists(model_path):
            clf = joblib.load(model_path)
            prob = float(clf.predict_proba(input_df)[0][1])
            
        pred_salary = None
        if os.path.exists(reg_model_path) and os.path.exists(reg_prep_path):
            reg_m = joblib.load(reg_model_path)
            reg_p = joblib.load(reg_prep_path)
            pred_sal_val = float(reg_m.predict(reg_p.transform(input_df))[0])
            pred_salary = max(0.0, round(pred_sal_val, 2))
            
        # Distance to Cluster centroids
        c1_dist = np.sqrt((cgpa - 5.39)**2 + ((coding - 26.18)/20)**2 + ((mock - 31.93)/20)**2)
        c2_dist = np.sqrt((cgpa - 7.74)**2 + ((coding - 65.98)/20)**2 + ((mock - 61.61)/20)**2)
        persona = "Cluster 2: Placement Ready" if c2_dist < c1_dist else "Cluster 1: At-Risk"
        
        # Company Tier Fit Analysis
        comp_score = (coding * 0.45) + (aptitude * 0.30) + (cgpa * 10 * 0.25)
        super_dream_pct = min(100, max(0, round((comp_score - 50) * 2.2, 1)))
        dream_pct = min(100, max(0, round((comp_score - 40) * 2.0, 1)))
        core_pct = min(100, max(0, round((comp_score - 30) * 1.8, 1)))
        
        # Actionable AI Coaching Tips
        tips = []
        if coding < 65:
            tips.append(f"Technical Uplift: Your Coding Score ({coding}%) is below the placed average (66%). Reaching 75%+ unlocks Super Dream technical rounds.")
        if mock < 62:
            tips.append(f"Interview Polish: Practice mock behavioral rounds. Raising mock score from {mock}% to 70%+ reduces recruiter rejection rates.")
        if internships == 0:
            tips.append("Industry Experience: Completing at least 1 internship adds direct leverage during salary negotiation.")
        if cgpa < 7.2:
            tips.append(f"Academic Baseline: CGPA is {cgpa}. Keeping CGPA >= 7.5 clears strict shortlisting cutoffs for top product companies.")
        if not tips:
            tips.append("Excellent Profile! You exceed the placement benchmark across all technical and academic dimensions. Aim for Super Dream offers.")
            
        return jsonify({
            "placement_probability": round(prob, 4),
            "placement_probability_pct": round(prob * 100, 1),
            "estimated_salary_lpa": pred_salary,
            "matched_persona": persona,
            "benchmarks": {
                "cgpa": 7.74,
                "coding_test_score": 65.98,
                "aptitude_test_score": 74.45,
                "mock_interview_score": 61.61,
                "attendance_percent": 81.03
            },
            "student_metrics": {
                "cgpa": cgpa,
                "coding_test_score": coding,
                "aptitude_test_score": aptitude,
                "mock_interview_score": mock,
                "attendance_percent": attendance
            },
            "company_fit": {
                "super_dream_readiness": super_dream_pct,
                "dream_readiness": dream_pct,
                "core_readiness": core_pct
            },
            "actionable_tips": tips
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/students/<int:student_id>")
def api_get_student_by_id(student_id):
    try:
        df = get_data()
        match = df[df["StudentID"] == student_id]
        if match.empty:
            return jsonify({"error": "Student not found"}), 404
        data = match.iloc[0].replace({np.nan: None}).to_dict()
        return jsonify({"student": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host=host, port=port, debug=debug)


