import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")

def get_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    
    if "PlacementStatus" in df.columns:
        df["PlacementStatus"] = df["PlacementStatus"].map({1: "Placed", 0: "Not Placed"}).fillna(df["PlacementStatus"])
        
    return df

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

@app.route("/api/plots")
def api_plots():
    try:
        plots_dir = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "plots")
        if not os.path.exists(plots_dir):
            return jsonify({"plots": []})

        plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        return jsonify({"plots": plots})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    plots_dir = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "plots")
    return send_from_directory(plots_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
