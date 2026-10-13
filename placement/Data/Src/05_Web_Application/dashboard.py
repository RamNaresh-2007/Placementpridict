import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Placement Analytics & ML Intelligence Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished cards and indicators
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .persona-card-risk {
        border-left: 5px solid #d62728;
        background-color: rgba(214, 39, 40, 0.08);
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .persona-card-ready {
        border-left: 5px solid #2ca02c;
        background-color: rgba(44, 160, 44, 0.08);
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .insight-badge {
        background: rgba(31, 119, 180, 0.15);
        color: #1f77b4;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
HIERARCHICAL_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "hierarchical")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Classification")
REGRESSION_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Regression")
PIPELINE_OUT = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs")
PCA_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca")
PCA_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "pca")
DBSCAN_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "dbscan")
ANOMALY_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection")
ANOMALY_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "anomaly_detection")
DATA_LEAKAGE_DIR = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "data_leakage")
DATA_LEAKAGE_OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "data_leakage")

# -----------------------------------------------------------------------------
# Data Caching & Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    if "PlacementStatus" in df.columns:
        df["PlacementStatusLabel"] = df["PlacementStatus"].map({1: "Placed", 0: "Not Placed"}).fillna(df["PlacementStatus"])
    return df

df = load_data()
if df.empty:
    st.stop()

# Benchmark Profile of Placed Students
placed_df = df[df["PlacementStatusLabel"] == "Placed"]
benchmarks = {
    "CGPA": round(placed_df["CGPA"].mean(), 2) if not placed_df.empty else 7.74,
    "CodingTestScore": round(placed_df["CodingTestScore"].mean(), 1) if not placed_df.empty else 66.0,
    "AptitudeTestScore": round(placed_df["AptitudeTestScore"].mean(), 1) if not placed_df.empty else 74.5,
    "MockInterviewScore": round(placed_df["MockInterviewScore"].mean(), 1) if not placed_df.empty else 61.6,
    "AttendancePercent": round(placed_df["AttendancePercent"].mean(), 1) if not placed_df.empty else 81.0,
}

# -----------------------------------------------------------------------------
# Global Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 Filter & Controls")
st.sidebar.markdown("Filter dataset across analytical views:")

# Gender Filter
gender_options = ["All"] + sorted(list(df["Gender"].dropna().unique()))
selected_gender = st.sidebar.selectbox("Gender", gender_options)

# Stream Filter
stream_options = ["All"] + sorted(list(df["Stream"].dropna().unique()))
selected_stream = st.sidebar.selectbox("Academic Stream", stream_options)

# College Tier Filter
tier_options = ["All"] + sorted(list(df["CollegeTier"].dropna().unique())) if "CollegeTier" in df.columns else ["All"]
selected_tier = st.sidebar.selectbox("College Tier", tier_options)

# History of Backlogs Filter
backlog_options = ["All"] + sorted(list(df["HistoryOfBacklogs"].dropna().unique())) if "HistoryOfBacklogs" in df.columns else ["All"]
selected_backlog = st.sidebar.selectbox("History of Backlogs", backlog_options)

# Placement Status Filter
placement_options = ["All"] + sorted(list(df["PlacementStatusLabel"].dropna().unique()))
selected_placement = st.sidebar.selectbox("Placement Status", placement_options)

# CGPA Range Slider
min_cgpa = float(df["CGPA"].min()) if "CGPA" in df.columns else 0.0
max_cgpa = float(df["CGPA"].max()) if "CGPA" in df.columns else 10.0
cgpa_range = st.sidebar.slider("CGPA Range", min_cgpa, max_cgpa, (min_cgpa, max_cgpa), 0.1)

# Apply filters
filtered_df = df.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_stream != "All":
    filtered_df = filtered_df[filtered_df["Stream"] == selected_stream]
if selected_tier != "All" and "CollegeTier" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["CollegeTier"] == selected_tier]
if selected_backlog != "All" and "HistoryOfBacklogs" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["HistoryOfBacklogs"] == selected_backlog]
if selected_placement != "All":
    filtered_df = filtered_df[filtered_df["PlacementStatusLabel"] == selected_placement]
if "CGPA" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["CGPA"] >= cgpa_range[0]) & (filtered_df["CGPA"] <= cgpa_range[1])]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** total students.")

# -----------------------------------------------------------------------------
# Main Header & Tab Navigation
# -----------------------------------------------------------------------------
st.title("🎓 Placement Analytics & Machine Learning Suite")
st.markdown("Interactive intelligence platform for student placement analytics, unsupervised hierarchical clustering, predictive what-if modeling, and salary regression.")

tabs = st.tabs([
    "📊 Analytics & Insights",
    "🌳 Unsupervised Clustering (Hierarchical & K-Means)",
    "📐 Dimensionality Reduction (PCA)",
    "🚨 Anomaly & Outlier Intelligence",
    "🛡️ Data Leakage & Validation Suite",
    "🧠 What-If Predictor & Salary Estimator",
    "📈 Models & Confusion Matrices",
    "🖼️ EDA & Generated Plots Gallery",
    "📋 Student Database Explorer"
])

# =============================================================================
# TAB 1: ANALYTICS & INSIGHTS
# =============================================================================
with tabs[0]:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_filtered = len(filtered_df)
    col1.metric("Students Analyzed", f"{total_filtered:,}")

    if total_filtered > 0:
        placed_count = len(filtered_df[filtered_df["PlacementStatusLabel"] == "Placed"])
        placement_rate = (placed_count / total_filtered) * 100
        col2.metric("Placement Rate", f"{placement_rate:.1f}%")
        
        avg_cgpa = filtered_df["CGPA"].mean() if "CGPA" in filtered_df.columns else 0
        col3.metric("Average CGPA", f"{avg_cgpa:.2f}")

        # Placed Average Salary
        avg_salary_text = "N/A"
        if "Salary Package" in filtered_df.columns:
            salary_numeric = pd.to_numeric(filtered_df["Salary Package"], errors='coerce')
            placed_salary = salary_numeric[(filtered_df["PlacementStatusLabel"] == "Placed") & (salary_numeric > 0)]
            if not placed_salary.empty:
                avg_salary_text = f"{placed_salary.mean():.2f} LPA" if placed_salary.mean() < 1000 else f"{placed_salary.mean()/100000:.2f} LPA"
        col4.metric("Avg Placed Package", avg_salary_text)

        cleaned_pct = ((total_filtered - filtered_df.isnull().any(axis=1).sum()) / total_filtered) * 100
        col5.metric("Complete Profiles", f"{cleaned_pct:.1f}%")

        st.markdown("---")

        # Row 1: CGPA Distribution & Placement Proportion
        st.subheader("Placement Distribution Dynamics")
        r1_col1, r1_col2 = st.columns([3, 2])

        with r1_col1:
            fig_cgpa = px.histogram(
                filtered_df,
                x="CGPA",
                nbins=35,
                color="PlacementStatusLabel",
                barmode="overlay",
                color_discrete_map={"Placed": "#2ca02c", "Not Placed": "#d62728"},
                opacity=0.75,
                title="CGPA Spread by Placement Status"
            )
            fig_cgpa.update_layout(bargap=0.08, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cgpa, use_container_width=True)

        with r1_col2:
            fig_pie = px.pie(
                filtered_df,
                names='PlacementStatusLabel',
                title='Overall Placement Ratio',
                hole=0.45,
                color='PlacementStatusLabel',
                color_discrete_map={"Placed": "#2ca02c", "Not Placed": "#d62728"}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # Row 2: Institutional & Academic Impact (Tier, Stream, Backlogs)
        st.subheader("Academic & Institutional Impact Breakdown")
        r2_col1, r2_col2 = st.columns(2)

        with r2_col1:
            if "CollegeTier" in filtered_df.columns:
                tier_grp = filtered_df.groupby("CollegeTier")["PlacementStatusLabel"].apply(lambda s: (s == "Placed").mean() * 100).reset_index(name="Placement Rate (%)")
                fig_tier = px.bar(
                    tier_grp,
                    x="CollegeTier",
                    y="Placement Rate (%)",
                    color="CollegeTier",
                    text_auto=".1f",
                    title="Placement Rate across College Tiers"
                )
                fig_tier.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_tier, use_container_width=True)

        with r2_col2:
            if "HistoryOfBacklogs" in filtered_df.columns:
                backlog_grp = filtered_df.groupby("HistoryOfBacklogs")["PlacementStatusLabel"].apply(lambda s: (s == "Placed").mean() * 100).reset_index(name="Placement Rate (%)")
                fig_backlog = px.bar(
                    backlog_grp,
                    x="HistoryOfBacklogs",
                    y="Placement Rate (%)",
                    color="HistoryOfBacklogs",
                    text_auto=".1f",
                    title="Impact of Backlog History on Placement Probability"
                )
                fig_backlog.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_backlog, use_container_width=True)

        st.markdown("---")

        # Row 3: Correlation Matrix & 3D Landscape
        st.subheader("Multi-Feature Interactions")
        r3_col1, r3_col2 = st.columns(2)

        with r3_col1:
            num_cols = ["CGPA", "CodingTestScore", "AptitudeTestScore", "MockInterviewScore", "AttendancePercent", "Internships", "Projects"]
            valid_num_cols = [c for c in num_cols if c in filtered_df.columns]
            if len(valid_num_cols) >= 2:
                corr = filtered_df[valid_num_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        with r3_col2:
            if all(c in filtered_df.columns for c in ["CGPA", "CodingTestScore", "AptitudeTestScore"]):
                sample_3d = filtered_df.sample(min(1200, len(filtered_df)), random_state=42)
                fig_3d = px.scatter_3d(
                    sample_3d,
                    x='CGPA', y='CodingTestScore', z='AptitudeTestScore',
                    color='PlacementStatusLabel',
                    opacity=0.65,
                    color_discrete_map={"Placed": "#2ca02c", "Not Placed": "#d62728"},
                    title='3D Placement Landscape (CGPA x Coding x Aptitude)'
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_3d, use_container_width=True)

        # Export Action
        st.markdown("---")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Current Filtered Slice (CSV)",
            data=csv,
            file_name="placement_analytics_filtered.csv",
            mime="text/csv"
        )
    else:
        st.warning("No records found matching the active sidebar filters. Please broaden your selection.")

# =============================================================================
# TAB 2: UNSUPERVISED CLUSTERING (HIERARCHICAL & K-MEANS)
# =============================================================================
with tabs[1]:
    st.header("🌳 Unsupervised Hierarchical & K-Means Clustering")
    st.markdown("""
    In contrast to supervised learning where models learn directly from labels, **Unsupervised Agglomerative Clustering** 
    discovers organic groupings solely based on student geometry and skill metrics—completely blind to `PlacementStatus`.
    """)

    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sampled Students (n)", "2,000", "Full O(N²) Tree")
    m2.metric("Optimal Clusters (K)", "2", "Silhouette = 0.4463")
    m3.metric("Best Distance Preserver", "Average Linkage", "Cophenetic = 0.7560")
    m4.metric("Ward Linkage Fidelity", "0.7298", "Minimizes Variance")

    st.markdown("---")

    # The 2 Discovered Personas
    st.subheader("Discovered Student Personas (Unsupervised)")
    st.markdown("Cutting Ward's agglomerative tree at optimal $K=2$ partitions the student body into two stark, distinct archetypes:")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("""
        <div class="persona-card-risk">
            <h3 style="color: #d62728; margin-top:0;">⚠️ Cluster 1: High-Risk Cohort</h3>
            <p><strong>Size:</strong> 402 Students (20.1% of cohort)</p>
            <p><strong>Observed Placement Rate:</strong> <span style="font-size: 1.3rem; font-weight: bold; color: #d62728;">5.72%</span></p>
            <ul>
                <li><strong>Mean CGPA:</strong> 5.39 / 10.0</li>
                <li><strong>Coding Test Score:</strong> 26.18% (Severe technical gap)</li>
                <li><strong>Mock Interview Score:</strong> 31.93% (Communication deficit)</li>
                <li><strong>Aptitude Test Score:</strong> 46.18%</li>
                <li><strong>Attendance:</strong> 60.71%</li>
            </ul>
            <p><em>Action Plan: Needs urgent technical remediation, coding bootcamps, and interview coaching.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("""
        <div class="persona-card-ready">
            <h3 style="color: #2ca02c; margin-top:0;">🌟 Cluster 2: Placement-Ready Cohort</h3>
            <p><strong>Size:</strong> 1,598 Students (79.9% of cohort)</p>
            <p><strong>Observed Placement Rate:</strong> <span style="font-size: 1.3rem; font-weight: bold; color: #2ca02c;">82.23%</span></p>
            <ul>
                <li><strong>Mean CGPA:</strong> 7.74 / 10.0</li>
                <li><strong>Coding Test Score:</strong> 65.98% (Strong proficiency)</li>
                <li><strong>Mock Interview Score:</strong> 61.61% (Job-ready presence)</li>
                <li><strong>Aptitude Test Score:</strong> 74.45%</li>
                <li><strong>Attendance:</strong> 81.03%</li>
            </ul>
            <p><em>Action Plan: High probability of recruitment; prime candidates for premium tier packages.</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Radar Comparison
    st.subheader("Skill Geometry: Cluster 1 vs Cluster 2 vs Benchmark")
    categories = ['CGPA (scaled x10)', 'Coding Score', 'Mock Interview', 'Aptitude Score', 'Attendance']
    
    fig_radar = go.Figure()
    # Cluster 1
    fig_radar.add_trace(go.Scatterpolar(
        r=[5.39*10, 26.18, 31.93, 46.18, 60.71],
        theta=categories,
        fill='toself',
        name='Cluster 1 (At-Risk, 5.7% Placed)',
        line_color='#d62728'
    ))
    # Cluster 2
    fig_radar.add_trace(go.Scatterpolar(
        r=[7.74*10, 65.98, 61.61, 74.45, 81.03],
        theta=categories,
        fill='toself',
        name='Cluster 2 (Ready, 82.2% Placed)',
        line_color='#2ca02c'
    ))
    # Benchmark
    fig_radar.add_trace(go.Scatterpolar(
        r=[benchmarks["CGPA"]*10, benchmarks["CodingTestScore"], benchmarks["MockInterviewScore"], benchmarks["AptitudeTestScore"], benchmarks["AttendancePercent"]],
        theta=categories,
        name='Benchmark (Placed Students)',
        line=dict(color='#1f77b4', dash='dash')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Multivariate Skill Radar: Archetypes vs Placement Benchmark"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Visual Artifacts Gallery
    st.subheader("Generated Hierarchical Clustering Visualizations")
    gal_col1, gal_col2 = st.columns(2)

    dendrogram_img = os.path.join(HIERARCHICAL_DIR, "dendrograms_by_linkage.png")
    silhouette_img = os.path.join(HIERARCHICAL_DIR, "silhouette_by_cut_height.png")
    profile_img = os.path.join(HIERARCHICAL_DIR, "cluster_placement_profile.png")

    with gal_col1:
        if os.path.exists(dendrogram_img):
            st.image(dendrogram_img, caption="Dendrograms by Linkage Method (Demonstrating Single-Linkage Chaining vs Ward Balance)", use_container_width=True)
        else:
            st.info("Run hierarchical_clustering.py to generate dendrogram visualizations.")

    with gal_col2:
        if os.path.exists(silhouette_img):
            st.image(silhouette_img, caption="Silhouette Score vs Cut Height (Confirming K=2 Peak)", use_container_width=True)
        else:
            st.info("Run hierarchical_clustering.py to generate silhouette score plots.")

    if os.path.exists(profile_img):
        st.image(profile_img, caption="External Outcome Validation: Placement Rate & Feature Space by Cluster", use_container_width=True)

    # Full Report Viewer
    report_file = os.path.join(HIERARCHICAL_DIR, "hierarchical_report.txt")
    if os.path.exists(report_file):
        with st.expander("📄 View Hierarchical Analytical Text Report (hierarchical_report.txt)"):
            with open(report_file, "r") as rf:
                st.code(rf.read(), language="text")

    st.markdown("---")
    st.subheader("DBSCAN Density-Based Spatial Clustering Visualizations")
    db_col1, db_col2 = st.columns(2)
    k_dist_img = os.path.join(DBSCAN_DIR, "k_distance_plot.png")
    db_pca_img = os.path.join(DBSCAN_DIR, "dbscan_clusters_pca_2d.png")

    with db_col1:
        if os.path.exists(k_dist_img):
            st.image(k_dist_img, caption="Optimal Epsilon Tuning: K-Distance Elbow Plot (eps = 2.88 knee)", use_container_width=True)
        else:
            st.info("k_distance_plot.png not found.")

    with db_col2:
        if os.path.exists(db_pca_img):
            st.image(db_pca_img, caption="DBSCAN 2D PCA Space: Core Dense Clusters vs Isolated Outlier Noise", use_container_width=True)
        else:
            st.info("dbscan_clusters_pca_2d.png not found.")

    db_report_file = os.path.join(DBSCAN_DIR, "dbscan_report.txt")
    if os.path.exists(db_report_file):
        with st.expander("📄 View DBSCAN Analytical Text Report (dbscan_report.txt)"):
            with open(db_report_file, "r") as drf:
                st.code(drf.read(), language="text")

# =============================================================================
# TAB 3: PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
with tabs[2]:
    st.header("📐 Principal Component Analysis (PCA) & Variance Exploration")
    st.markdown("""
    Unsupervised orthogonal dimensionality reduction projecting 20 academic, assessment, and activity features into 
    uncorrelated variance axes. PCA discovers that **76.1%** of all student geometric variance is captured by just the first 2 principal components.
    """)

    # Load precomputed PCA summary if available
    summary_path = os.path.join(PCA_DIR, "pca_summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(PCA_OUTPUTS_DIR, "pca_summary.json")
    
    pca_summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                pca_summary = json.load(f)
        except Exception:
            pass

    pc1_var = pca_summary.get("pc1_variance_pct", 64.24) if pca_summary else 64.24
    pc2_var = pca_summary.get("pc2_variance_pct", 11.84) if pca_summary else 11.84
    cum_2d = pca_summary.get("pc1_pc2_cumulative_pct", 76.08) if pca_summary else 76.08

    # Overview KPI row
    pca_c1, pca_c2, pca_c3, pca_c4, pca_c5 = st.columns(5)
    pca_c1.metric("PC1 Variance", f"{pc1_var:.1f}%", "Academic History Axis")
    pca_c2.metric("PC2 Variance", f"{pc2_var:.1f}%", "Practical Skills Axis")
    pca_c3.metric("2D Total Coverage", f"{cum_2d:.1f}%", "PC1 + PC2 Spread")
    pca_c4.metric("80% Threshold", "3 Comps", "85% Dimensions Discarded")
    pca_c5.metric("95% Threshold", "10 Comps", "Half Dimensions Retained")

    st.markdown("---")

    # Interactive Plots Row: Scree Plot & 2D PCA Projection
    st.subheader("Interactive Geometric Projections")
    r1_pca1, r1_pca2 = st.columns(2)

    with r1_pca1:
        # Scree & Cumulative Variance Chart
        if pca_summary and "variance_by_component" in pca_summary:
            vbc = pca_summary["variance_by_component"]
            comp_labels = [c["component"] for c in vbc]
            var_vals = [c["variance_pct"] for c in vbc]
            cum_vals = [c["cumulative_pct"] for c in vbc]
        else:
            comp_labels = [f"PC{i+1}" for i in range(20)]
            var_vals = [64.24, 11.84, 4.45, 3.26, 2.67, 2.41, 2.04, 1.84, 1.72, 1.57, 1.33, 1.16, 0.32, 0.18, 0.18, 0.17, 0.17, 0.16, 0.15, 0.14]
            cum_vals = list(np.cumsum(var_vals))

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=comp_labels,
            y=var_vals,
            name="Individual Variance (%)",
            marker_color="#6366f1",
            opacity=0.85
        ))
        fig_scree.add_trace(go.Scatter(
            x=comp_labels,
            y=cum_vals,
            name="Cumulative Variance (%)",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=6)
        ))
        fig_scree.add_hline(y=80, line_dash="dash", line_color="#0ea5e9", annotation_text="80% Cutoff (PC3)", annotation_position="top left")
        fig_scree.add_hline(y=95, line_dash="dash", line_color="#f59e0b", annotation_text="95% Cutoff (PC10)", annotation_position="top left")
        fig_scree.update_layout(
            title="Variance Scree & Cumulative Retention Curve",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained (%)",
            legend=dict(x=0.6, y=0.35),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    with r1_pca2:
        # 2D Interactive PCA Projection of Current Filtered Dataset
        num_features = [
            "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
            "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
            "CGPA", "AttendancePercent", "Internships", "Projects",
            "Workshops", "Certifications", "Publications", "AptitudeTestScore",
            "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "ExtraCurricular"
        ]
        valid_features = [f for f in num_features if f in filtered_df.columns]

        if len(valid_features) >= 5 and len(filtered_df) > 10:
            sample_n = min(1500, len(filtered_df))
            sample_pca_df = filtered_df.sample(sample_n, random_state=42)
            imputed_sample = sample_pca_df[valid_features].fillna(sample_pca_df[valid_features].median())
            scaler_pca = StandardScaler()
            scaled_sample = scaler_pca.fit_transform(imputed_sample)
            
            pca_model = PCA(n_components=2, random_state=42)
            coords_2d = pca_model.fit_transform(scaled_sample)
            
            sample_pca_df = sample_pca_df.copy()
            sample_pca_df["PC1"] = coords_2d[:, 0]
            sample_pca_df["PC2"] = coords_2d[:, 1]
            
            fig_pca_scatter = px.scatter(
                sample_pca_df,
                x="PC1",
                y="PC2",
                color="PlacementStatusLabel",
                hover_data=["StudentID", "CGPA", "CodingTestScore", "MockInterviewScore", "AttendancePercent"],
                color_discrete_map={"Placed": "#2ca02c", "Not Placed": "#d62728"},
                opacity=0.7,
                title=f"2D Student Landscape (Sample: {sample_n:,} Students)"
            )
            fig_pca_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pca_scatter, use_container_width=True)
        else:
            st.info("Insufficient numeric features or samples to construct interactive 2D PCA projection.")

    st.markdown("---")

    # Row 2: Latent Vector Interpretation & Loadings Analysis
    st.subheader("Latent Feature Interpretations & Factor Loadings")
    lat_col1, lat_col2 = st.columns(2)

    with lat_col1:
        st.markdown("""
        <div class="persona-card-ready">
            <h4 style="color: #4f46e5; margin-top:0;">🎓 PC1: Academic Performance Factor (64.2% Variance)</h4>
            <p>PC1 represents sustained academic consistency over time. Semester grades (SGPA Sem 2 through 8) 
            and CGPA load with almost identical positive weights (<strong>+0.254 to +0.263</strong>). 
            This single composite axis accounts for nearly two-thirds of all student variance.</p>
            <p><strong>Primary Contributors:</strong> <code>SGPA_Sem8 (+0.263)</code>, <code>SGPA_Sem7 (+0.262)</code>, <code>SGPA_Sem6 (+0.260)</code>, <code>CGPA (+0.254)</code></p>
        </div>
        <div class="persona-card-risk" style="border-left-color: #8b5cf6; background-color: rgba(139, 92, 246, 0.08);">
            <h4 style="color: #8b5cf6; margin-top:0;">💻 PC2: Applied Technical & Problem-Solving Factor (11.8% Variance)</h4>
            <p>PC2 isolates practical candidate readiness independent of grades: technical tests, coding agility, 
            and hands-on portfolio experience load strongly positive, while early introductory performance loads negative.</p>
            <p><strong>Primary Contributors:</strong> <code>CodingTestScore (+0.271)</code>, <code>AptitudeTestScore (+0.263)</code>, <code>Projects (+0.258)</code>, <code>Internships (+0.246)</code> vs <code>SGPA_Sem1 (-0.245)</code></p>
        </div>
        """, unsafe_allow_html=True)

    with lat_col2:
        top_pc1 = pca_summary.get("top_pc1_loadings", [
            {"feature": "SGPA_Sem8", "loading": 0.263},
            {"feature": "SGPA_Sem7", "loading": 0.262},
            {"feature": "SGPA_Sem6", "loading": 0.260},
            {"feature": "SGPA_Sem5", "loading": 0.259},
            {"feature": "SGPA_Sem4", "loading": 0.258},
            {"feature": "CGPA", "loading": 0.254}
        ]) if pca_summary else []
        top_pc2 = pca_summary.get("top_pc2_loadings", [
            {"feature": "CodingTestScore", "loading": 0.271},
            {"feature": "AptitudeTestScore", "loading": 0.263},
            {"feature": "Projects", "loading": 0.258},
            {"feature": "Internships", "loading": 0.246},
            {"feature": "SGPA_Sem1", "loading": -0.245},
            {"feature": "Certifications", "loading": 0.234}
        ]) if pca_summary else []

        tab_l1, tab_l2 = st.tabs(["Top PC1 Loadings", "Top PC2 Loadings"])
        with tab_l1:
            st.dataframe(pd.DataFrame(top_pc1), use_container_width=True)
        with tab_l2:
            st.dataframe(pd.DataFrame(top_pc2), use_container_width=True)

    st.markdown("---")

    # High-Resolution Diagnostic Artifacts
    st.subheader("Generated PCA Diagnostic Visualizations")
    art_col1, art_col2, art_col3 = st.columns(3)

    biplot_img = os.path.join(PCA_DIR, "loadings_biplot.png")
    if not os.path.exists(biplot_img):
        biplot_img = os.path.join(PCA_OUTPUTS_DIR, "loadings_biplot.png")

    scatter_img = os.path.join(PCA_DIR, "pca_2d_by_placement.png")
    if not os.path.exists(scatter_img):
        scatter_img = os.path.join(PCA_OUTPUTS_DIR, "pca_2d_by_placement.png")

    scree_img = os.path.join(PCA_DIR, "scree_and_cumulative_variance.png")
    if not os.path.exists(scree_img):
        scree_img = os.path.join(PCA_OUTPUTS_DIR, "scree_and_cumulative_variance.png")

    with art_col1:
        if os.path.exists(scree_img):
            st.image(scree_img, caption="1. Scree & Cumulative Variance Curve (76.1% in 2D)", use_container_width=True)
        else:
            st.info("Scree plot not found.")

    with art_col2:
        if os.path.exists(biplot_img):
            st.image(biplot_img, caption="2. Feature Loadings Biplot: 20 Numeric Vectors in PC1 vs PC2", use_container_width=True)
        else:
            st.info("Loadings biplot not found.")

    with art_col3:
        if os.path.exists(scatter_img):
            st.image(scatter_img, caption="3. All 35,000 Students Projected into PC1/PC2 Coordinate Space", use_container_width=True)
        else:
            st.info("2D scatter plot not found.")

    # Full Analytical Report Viewer
    pca_rep_file = os.path.join(PCA_DIR, "pca_report.txt")
    if not os.path.exists(pca_rep_file):
        pca_rep_file = os.path.join(PCA_OUTPUTS_DIR, "pca_report.txt")
    if os.path.exists(pca_rep_file):
        with st.expander("📄 View Full Analytical Text Report (pca_report.txt)"):
            with open(pca_rep_file, "r") as rf:
                st.code(rf.read(), language="text")

# =============================================================================
# TAB 4: ANOMALY & OUTLIER INTELLIGENCE (SESSION 27)
# =============================================================================
with tabs[3]:
    st.header("🚨 Session 27: Unsupervised Anomaly & Outlier Intelligence Suite")
    st.markdown("Multi-model unsupervised detection isolating multidimensional applicant contradictions via tree-partitioning (Isolation Forest), robust covariance (Elliptic Envelope), and density spatial filtering (DBSCAN).")

    # Load summary JSON if available
    anomaly_summary = None
    summary_path = os.path.join(ANOMALY_DIR, "anomaly_summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(ANOMALY_OUTPUTS_DIR, "anomaly_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                anomaly_summary = json.load(f)
        except Exception:
            pass

    # Top KPI Metrics Row
    anom_c1, anom_c2, anom_c3, anom_c4, anom_c5 = st.columns(5)
    anom_c1.metric("Elliptic Envelope ROC", "98.8%", "FastMCD Covariance")
    anom_c2.metric("True Anomaly Recall", "91.6%", "Ground Truth Captured")
    anom_c3.metric("Isolation Forest ROC", "93.1%", "Full 35k Cohort O(n log n)")
    anom_c4.metric("DBSCAN Precision", "93.9%", "Pure Outlier Noise")
    anom_c5.metric("Detected Outlier Pool", "1,225", "3.5% Contamination Prior")

    st.markdown("---")

    # Multi-Algorithm Benchmark Comparison Table
    st.subheader("Multi-Algorithm Anomaly Detection Benchmark")
    if anomaly_summary and "benchmark_comparison" in anomaly_summary:
        bench_df = pd.DataFrame(anomaly_summary["benchmark_comparison"])
        bench_display = bench_df.rename(columns={
            "model": "Algorithm",
            "paradigm": "Mathematical Paradigm",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1_score": "F1-Score",
            "strengths": "Empirical Advantage"
        })
        st.dataframe(bench_display, use_container_width=True)
    else:
        st.info("Loading benchmark metrics...")

    st.markdown("---")

    # Diagnostic Visualizations Gallery
    st.subheader("Generated Diagnostic Visualizations")
    anom_col1, anom_col2 = st.columns(2)

    plot1_path = os.path.join(ANOMALY_DIR, "01_anomaly_score_distribution.png")
    if not os.path.exists(plot1_path):
        plot1_path = os.path.join(ANOMALY_OUTPUTS_DIR, "01_anomaly_score_distribution.png")

    plot2_path = os.path.join(ANOMALY_DIR, "02_pca_2d_anomaly_landscape.png")
    if not os.path.exists(plot2_path):
        plot2_path = os.path.join(ANOMALY_OUTPUTS_DIR, "02_pca_2d_anomaly_landscape.png")

    plot3_path = os.path.join(ANOMALY_DIR, "03_algorithm_comparison_benchmark.png")
    if not os.path.exists(plot3_path):
        plot3_path = os.path.join(ANOMALY_OUTPUTS_DIR, "03_algorithm_comparison_benchmark.png")

    plot4_path = os.path.join(ANOMALY_DIR, "04_anomaly_feature_profiles.png")
    if not os.path.exists(plot4_path):
        plot4_path = os.path.join(ANOMALY_OUTPUTS_DIR, "04_anomaly_feature_profiles.png")

    plot5_path = os.path.join(ANOMALY_DIR, "05_contamination_sensitivity.png")
    if not os.path.exists(plot5_path):
        plot5_path = os.path.join(ANOMALY_OUTPUTS_DIR, "05_contamination_sensitivity.png")

    with anom_col1:
        if os.path.exists(plot1_path):
            st.image(plot1_path, caption="1. Anomaly Decision Score Distribution & 3.5% Cutoff Threshold", use_container_width=True)
        if os.path.exists(plot3_path):
            st.image(plot3_path, caption="3. 5-Algorithm Detection Performance Benchmark", use_container_width=True)
        if os.path.exists(plot5_path):
            st.image(plot5_path, caption="5. Contamination Sensitivity Curve (1% to 10% Sweep)", use_container_width=True)

    with anom_col2:
        if os.path.exists(plot2_path):
            st.image(plot2_path, caption="2. 2D PCA Anomaly Landscape (Inliers vs Flagged vs True Positives)", use_container_width=True)
        if os.path.exists(plot4_path):
            st.image(plot4_path, caption="4. Feature Deviation Archetype (Z-Score Shift from Normal Students)", use_container_width=True)

    st.markdown("---")

    # Interactive Candidate Anomaly Inspector
    st.subheader("Interactive Candidate Anomaly Inspector")
    st.markdown("Inspect applicant records flagged as anomalies vs typical candidate profiles to observe multidimensional covariance contradictions.")
    
    if "IsAnomaly" in df.columns:
        filter_choice = st.radio("Cohort Selection:", ["Ground Truth Anomalies (IsAnomaly=1)", "Normal Students (IsAnomaly=0)", "All Students"], horizontal=True)
        if filter_choice == "Ground Truth Anomalies (IsAnomaly=1)":
            insp_df = df[df["IsAnomaly"] == 1]
        elif filter_choice == "Normal Students (IsAnomaly=0)":
            insp_df = df[df["IsAnomaly"] == 0]
        else:
            insp_df = df

        display_cols = ["StudentID", "CGPA", "AttendancePercent", "Projects", "Internships", "CodingTestScore", "AptitudeTestScore", "PlacementStatus"]
        if "Salary Package" in df.columns:
            display_cols.append("Salary Package")
        
        st.dataframe(insp_df[display_cols].head(50), use_container_width=True)
        st.caption(f"Showing sample of {len(insp_df):,} matching candidates.")

    # Analytical Report Viewer
    anom_rep_file = os.path.join(ANOMALY_DIR, "anomaly_report.txt")
    if not os.path.exists(anom_rep_file):
        anom_rep_file = os.path.join(ANOMALY_OUTPUTS_DIR, "anomaly_report.txt")
    if os.path.exists(anom_rep_file):
        with st.expander("📄 View Full Analytical Text Report (anomaly_report.txt)"):
            with open(anom_rep_file, "r", encoding="utf-8") as rf:
                st.code(rf.read(), language="text")

# =============================================================================
# TAB 5: DATA LEAKAGE PREVENTION & VALIDATION ARCHITECTURE (SESSION 28)
# =============================================================================
with tabs[4]:
    st.header("🛡️ Session 28: Data Leakage Prevention & Robust Model Validation Suite")
    st.markdown("Guaranteed generalization integrity via scikit-learn `Pipeline` encapsulation, strictly held-out vault test sets, 5×3 Nested Cross-Validation, and empirical proof of distribution contamination.")

    # Load leakage summary JSON
    leakage_summary = None
    summary_path = os.path.join(DATA_LEAKAGE_DIR, "leakage_summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(DATA_LEAKAGE_OUTPUTS_DIR, "leakage_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                leakage_summary = json.load(f)
        except Exception:
            pass

    # Top KPI Metrics Row
    l_c1, l_c2, l_c3, l_c4, l_c5 = st.columns(5)
    l_c1.metric("Vault Test Accuracy", "89.5%", "1,000 Isolated Students")
    l_c2.metric("Vault Test ROC-AUC", "94.1%", "Honest Out-of-Sample ROC")
    l_c3.metric("Nested CV F1-Score", "92.1% ± 0.6%", "5 Outer Folds (Zero Snooping)")
    l_c4.metric("Tuning Snooping Delta", "+0.07%", "Isolated via Nested CV")
    l_c5.metric("Noise Selection Proof", "50.5% vs 68.0%", "Hastie Baseline vs Leaky Illusion")

    st.markdown("---")

    # Architecture Overview Cards
    st.subheader("Architectural Workflows & Leakage Mitigation Strategy")
    arch_c1, arch_c2, arch_c3 = st.columns(3)

    with arch_c1:
        st.error("⚠️ The Fatal Flaw: Leaky Pipeline")
        st.markdown("""
        - **Anti-Pattern**: Fitting `StandardScaler().fit_transform(X)` before splitting data.
        - **Flaw**: Global calculation of $\\mu$ and $\\sigma$ contaminates training partition with test information.
        - **Catastrophic Failure**: Feature selection on pure Gaussian random noise invents a **68.0% accuracy illusion** out of thin air.
        """)

    with arch_c2:
        st.success("🛡️ Workflow 1: Vault-Lock Pipeline (Image 2)")
        st.markdown("""
        - **Quarantine**: Immediately isolate 20% test partition (1,000 students) before any processing.
        - **Pipeline Encapsulation**: Embed `StandardScaler` and `LogisticRegression` within `Pipeline`.
        - **Dev CV**: 5-Fold Stratified CV (F1: 92.2% ± 1.2%, AUC: 95.1% ± 0.5%).
        - **Vault Certification**: Final evaluation once on untouched vault test set (89.5% Acc, 94.1% AUC).
        """)

    with arch_c3:
        st.info("🔄 Workflow 2: 5×3 Nested CV (Image 1)")
        st.markdown("""
        - **Outer Loop (5 Folds)**: Estimates true generalization performance.
        - **Inner Loop (3 Folds)**: Optimizes hyperparameter $C \\in [0.001, 100]$ strictly within training folds.
        - **Result**: Completely eliminates **hyperparameter tuning bias** (Unbiased Nested CV F1: 92.13% vs Non-nested 92.21%).
        """)

    st.markdown("---")

    # Interactive Sub-Tabs for Details
    l_subtabs = st.tabs([
        "📊 Benchmark Comparison",
        "🔄 5-Fold Nested CV Details",
        "🔬 Distribution Contamination Proof",
        "🖼️ Generated Publication Visualizations",
        "📄 Diagnostic Text Report"
    ])

    with l_subtabs[0]:
        st.subheader("Leaky vs. Leakage-Free Pipeline Performance Benchmark")
        st.markdown("5-Fold Stratified Cross-Validation on $N=5,000$ Student Dataset:")

        benchmark_table = pd.DataFrame([
            {"Metric": "Accuracy", "Leaky (Contaminated)": "0.8954 ± 0.0135", "Leakage-Free (Pipeline)": "0.8954 ± 0.0135", "Optimism Gap": "+0.0000", "Status": "✅ Generalizes"},
            {"Metric": "Precision", "Leaky (Contaminated)": "0.9067 ± 0.0111", "Leakage-Free (Pipeline)": "0.9067 ± 0.0111", "Optimism Gap": "+0.0000", "Status": "✅ Generalizes"},
            {"Metric": "Recall", "Leaky (Contaminated)": "0.9363 ± 0.0210", "Leakage-Free (Pipeline)": "0.9363 ± 0.0210", "Optimism Gap": "+0.0000", "Status": "✅ Generalizes"},
            {"Metric": "F1-Score", "Leaky (Contaminated)": "0.9212 ± 0.0107", "Leakage-Free (Pipeline)": "0.9212 ± 0.0107", "Optimism Gap": "+0.0000", "Status": "✅ Generalizes"},
            {"Metric": "ROC-AUC", "Leaky (Contaminated)": "0.9487 ± 0.0178", "Leakage-Free (Pipeline)": "0.9487 ± 0.0178", "Optimism Gap": "+0.0000", "Status": "✅ Generalizes"},
            {"Metric": "Noise Feature Selection (Hastie Proof)", "Leaky (Contaminated)": "0.6800 (68.0%)", "Leakage-Free (Pipeline)": "0.5050 (50.5%)", "Optimism Gap": "+0.1750 (+17.5%)", "Status": "🚨 Fatal Illusion"}
        ])
        st.dataframe(benchmark_table, use_container_width=True)

        st.caption("Hastie & Tibshirani Feature Selection Experiment: 1,000 samples of pure standard Gaussian noise with binary random targets. Pre-split feature selection picks top 10 noise features correlated with targets by chance, fabricating an apparent 68.0% accuracy illusion.")

    with l_subtabs[1]:
        st.subheader("5×3 Nested Cross-Validation Outer Fold Performance")
        if leakage_summary and "nested_cv" in leakage_summary and "fold_details" in leakage_summary["nested_cv"]:
            folds_data = pd.DataFrame(leakage_summary["nested_cv"]["fold_details"])
            folds_data = folds_data.rename(columns={
                "fold": "Outer Fold",
                "best_C": "Best Regularization (C)",
                "inner_cv_best_f1": "Inner CV Best F1",
                "outer_val_f1": "Outer Validation F1",
                "outer_val_acc": "Outer Validation Accuracy"
            })
            st.dataframe(folds_data, use_container_width=True)
            st.markdown(f"**Unbiased Nested CV F1**: `{leakage_summary['nested_cv'].get('nested_f1_mean', 0.9213):.4f} ± {leakage_summary['nested_cv'].get('nested_f1_std', 0.0058):.4f}` | **Non-Nested Best CV**: `{leakage_summary['nested_cv'].get('non_nested_optimistic_f1', 0.9221):.4f}` (Tuning Snooping Delta: `+{leakage_summary['nested_cv'].get('tuning_optimism_delta', 0.0007):.4f}`)")
        else:
            st.info("Loading fold details...")

    with l_subtabs[2]:
        st.subheader("Empirical Distribution Contamination Shift Proof")
        st.markdown("Demonstrating how test instances distort $\\mu$ and $\\sigma$ parameter estimates when standardized globally:")
        if leakage_summary and "contamination" in leakage_summary:
            contam_df = pd.DataFrame(leakage_summary["contamination"])
            st.dataframe(contam_df, use_container_width=True)
        else:
            st.info("Distribution contamination summary available in report.")

    with l_subtabs[3]:
        st.subheader("Generated Diagnostic Visualizations Gallery")
        plot_files = [
            ("01_leakage_concept_architecture.png", "Architectural Pipeline Diagram (Fatal Flaw vs Vault Lock vs Nested CV)"),
            ("02_leaky_vs_leakage_free_benchmark.png", "Leaky vs. Leakage-Free Benchmark & Hastie/Tibshirani Noise Proof"),
            ("03_distribution_contamination_proof.png", "Empirical Distribution Contamination Proof (Mu & Sigma Shifts)"),
            ("04_nested_cv_outer_vs_inner_performance.png", "Nested CV Outer Fold vs Inner Grid Search Performance"),
            ("05_vault_lock_test_evaluation.png", "Vault-Lock Test Set Evaluation, Confusion Matrix & ROC Curve")
        ]

        plot_cols = st.columns(2)
        for idx, (pfile, pcaption) in enumerate(plot_files):
            col = plot_cols[idx % 2]
            p_path = os.path.join(DATA_LEAKAGE_DIR, pfile)
            if not os.path.exists(p_path):
                p_path = os.path.join(DATA_LEAKAGE_OUTPUTS_DIR, pfile)
            if os.path.exists(p_path):
                with col:
                    st.image(p_path, caption=pcaption, use_container_width=True)
                    st.markdown("---")

    with l_subtabs[4]:
        st.subheader("Session 28 Full Diagnostic Text Report")
        l_rep_file = os.path.join(DATA_LEAKAGE_DIR, "leakage_report.txt")
        if not os.path.exists(l_rep_file):
            l_rep_file = os.path.join(DATA_LEAKAGE_OUTPUTS_DIR, "leakage_report.txt")
        if os.path.exists(l_rep_file):
            with open(l_rep_file, "r", encoding="utf-8") as rf:
                st.code(rf.read(), language="text")
        else:
            st.info("Report file not found.")

# =============================================================================
# TAB 6: WHAT-IF PREDICTOR & DUAL SALARY ESTIMATOR
# =============================================================================
with tabs[5]:
    st.header("🧠 Dual AI Predictor: Placement Probability & Salary Package")
    st.markdown("Tune a student profile below to obtain live classification probability, expected salary package, and targeted profile improvement tips.")

    input_col, output_col = st.columns([1, 1])

    with input_col:
        st.subheader("Student Skill Profile")
        
        # Classifier selector
        available_models = {
            "Logistic Regression (Pipeline)": "logistic_regression.pkl",
            "Random Forest Classifier": "random_forest.pkl",
            "Gradient Boosting Classifier": "gradient_boosting.pkl",
            "Decision Tree Classifier": "decision_tree.pkl",
            "K-Nearest Neighbors (KNN)": "knn.pkl"
        }
        selected_model_name = st.selectbox("Choose Classification Engine", list(available_models.keys()))
        selected_model_file = available_models[selected_model_name]

        sim_cgpa = st.slider("CGPA (0 - 10)", 5.0, 10.0, 7.5, 0.1)
        sim_coding = st.slider("Coding Test Score (%)", 0, 100, 65, 1)
        sim_aptitude = st.slider("Aptitude Test Score (%)", 0, 100, 70, 1)
        sim_mock = st.slider("Mock Interview Score (%)", 0, 100, 65, 1)
        sim_attendance = st.slider("Attendance Rate (%)", 40, 100, 85, 1)

        with st.expander("💼 Experience & Institutional Parameters", expanded=True):
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                sim_internships = st.number_input("Internships", min_value=0, max_value=5, value=1)
                sim_projects = st.number_input("Projects", min_value=0, max_value=10, value=2)
                sim_certs = st.number_input("Certifications", min_value=0, max_value=10, value=1)
            with exp_col2:
                sim_stream = st.selectbox("Stream", ["CS", "IT", "ECE", "Mechanical", "EE", "Civil"], index=0)
                sim_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
                sim_backlogs = st.selectbox("History of Backlogs", ["No", "Yes"], index=0)
                sim_gender = st.selectbox("Gender", ["Male", "Female"], index=0)

    with output_col:
        st.subheader("Live Model Inference")

        model_path = os.path.join(CLASSIFICATION_DIR, selected_model_file)
        prep_path = os.path.join(CLASSIFICATION_DIR, "standard_preprocessor.pkl")
        reg_model_path = os.path.join(REGRESSION_DIR, "linear_regression.pkl")
        reg_prep_path = os.path.join(REGRESSION_DIR, "regression_preprocessor.pkl")

        # Build inference input dataframe with all 27 features
        input_dict = {
            'SGPA_Sem1': [sim_cgpa], 'SGPA_Sem2': [sim_cgpa], 'SGPA_Sem3': [sim_cgpa], 'SGPA_Sem4': [sim_cgpa],
            'SGPA_Sem5': [sim_cgpa], 'SGPA_Sem6': [sim_cgpa], 'SGPA_Sem7': [sim_cgpa], 'SGPA_Sem8': [sim_cgpa],
            'CGPA': [sim_cgpa], 'AttendancePercent': [sim_attendance],
            'Internships': [sim_internships], 'Projects': [sim_projects], 
            'Workshops': [0], 'Certifications': [sim_certs], 'Publications': [0],
            'AptitudeTestScore': [sim_aptitude], 'SoftSkillsRating': [80.0],
            'CodingTestScore': [sim_coding], 'MockInterviewScore': [sim_mock], 'ExtraCurricular': [1],
            'Gender': [sim_gender], 'City': ['Pune'], 'CollegeTier': [sim_tier.replace(" ", "")],
            'Stream': [sim_stream], 'Specialisation': ['AI'], 'Hostel': ['No'], 'HistoryOfBacklogs': [sim_backlogs]
        }
        input_df = pd.DataFrame(input_dict)

        if os.path.exists(model_path):
            try:
                clf = joblib.load(model_path)
                
                # Check if pipeline or raw model
                if hasattr(clf, "steps"):
                    prob = clf.predict_proba(input_df)[0][1]
                elif os.path.exists(prep_path):
                    preprocessor = joblib.load(prep_path)
                    X_transformed = preprocessor.transform(input_df)
                    prob = clf.predict_proba(X_transformed)[0][1]
                else:
                    prob = 0.5
                
                prob_pct = prob * 100

                # 1. Placement Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={'suffix': "%", 'font': {'size': 32}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 45], 'color': "rgba(214, 39, 40, 0.3)"},
                            {'range': [45, 75], 'color': "rgba(255, 127, 14, 0.3)"},
                            {'range': [75, 100], 'color': "rgba(44, 160, 44, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob_pct
                        }
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # 2. Predicted Salary Package (Regression)
                pred_salary_text = "N/A"
                if os.path.exists(reg_model_path) and os.path.exists(reg_prep_path):
                    reg_model = joblib.load(reg_model_path)
                    reg_prep = joblib.load(reg_prep_path)
                    reg_X = reg_prep.transform(input_df)
                    pred_sal = reg_model.predict(reg_X)[0]
                    if pred_sal > 0:
                        pred_salary_text = f"{pred_sal:.2f} LPA" if pred_sal < 100 else f"{pred_sal/100000:.2f} LPA"

                # Status Banners
                if prob >= 0.75:
                    st.success(f"🎉 **High Placement Likelihood ({prob_pct:.1f}%)** | Estimated Salary: **{pred_salary_text}**")
                elif prob >= 0.45:
                    st.warning(f"⚠️ **Borderline Placement Odds ({prob_pct:.1f}%)** | Estimated Salary: **{pred_salary_text}**")
                else:
                    st.error(f"📉 **At-Risk Profile ({prob_pct:.1f}%)** | High risk of non-placement")

                # Persona Matching
                # Distance to Cluster 1 centroid vs Cluster 2 centroid
                c1_dist = np.sqrt((sim_cgpa - 5.39)**2 + ((sim_coding - 26.18)/20)**2 + ((sim_mock - 31.93)/20)**2)
                c2_dist = np.sqrt((sim_cgpa - 7.74)**2 + ((sim_coding - 65.98)/20)**2 + ((sim_mock - 61.61)/20)**2)
                assigned_cluster = "Cluster 2 (Placement Ready)" if c2_dist < c1_dist else "Cluster 1 (At-Risk)"
                st.info(f"🏷️ **Matched Unsupervised Persona**: `{assigned_cluster}`")

                # Smart Marginal Gain Simulation
                st.markdown("#### 💡 AI Marginal Gain Recommendations")
                
                # Test Coding +15
                input_test = input_df.copy()
                input_test['CodingTestScore'] = [min(100, sim_coding + 15)]
                if hasattr(clf, "steps"):
                    prob_coding_boost = clf.predict_proba(input_test)[0][1]
                else:
                    prob_coding_boost = clf.predict_proba(preprocessor.transform(input_test))[0][1]
                coding_delta = (prob_coding_boost - prob) * 100

                # Test Internship +1
                input_test2 = input_df.copy()
                input_test2['Internships'] = [sim_internships + 1]
                if hasattr(clf, "steps"):
                    prob_intern_boost = clf.predict_proba(input_test2)[0][1]
                else:
                    prob_intern_boost = clf.predict_proba(preprocessor.transform(input_test2))[0][1]
                intern_delta = (prob_intern_boost - prob) * 100

                st.write(f"- **Coding Proficiency Boost**: Improving Coding Score from {sim_coding}% to {min(100, sim_coding+15)}% shifts probability by **{coding_delta:+.1f}%**.")
                st.write(f"- **Internship Experience**: Completing +1 additional internship adds **{intern_delta:+.1f}%** to your probability.")

            except Exception as e:
                st.error(f"Error during model evaluation: {e}")
        else:
            st.info(f"Model file `{selected_model_file}` not found at `{CLASSIFICATION_DIR}`.")

# =============================================================================
# TAB 7: MODELS, BENCHMARKS & CONFUSION MATRICES
# =============================================================================
with tabs[6]:
    st.header("📈 Model Benchmarks & Diagnostic Evaluations")

    # Section 1: Scaling Impact Comparison
    st.subheader("1. Impact of Feature Scaling across Model Families")
    scaling_csv = os.path.join(BASE_DIR, "..", "..", "Outputs", "model_scaling_results.csv")
    if not os.path.exists(scaling_csv):
        scaling_csv = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "Outputs", "model_scaling_results.csv")
    
    if os.path.exists(scaling_csv):
        scaling_df = pd.read_csv(scaling_csv)
        st.dataframe(scaling_df, use_container_width=True)
        
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            fig_sc = px.bar(
                scaling_df, x="Model Name", y="Score", color="Scaled",
                barmode="group", facet_col="Task",
                title="Model Score: Scaled vs Unscaled"
            )
            fig_sc.update_yaxes(matches=None)
            st.plotly_chart(fig_sc, use_container_width=True)
        with sc_col2:
            fig_time = px.bar(
                scaling_df, x="Model Name", y="Train Time (s)", color="Scaled",
                barmode="group", facet_col="Task",
                title="Training Time (s): Scaled vs Unscaled"
            )
            fig_time.update_yaxes(matches=None)
            st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # Section 2: Classification Leaderboard & Confusion Matrices
    st.subheader("2. Supervised Classification Leaderboard")
    class_summary = os.path.join(CLASSIFICATION_DIR, "classification_model_comparison.csv")
    if os.path.exists(class_summary):
        cdf = pd.read_csv(class_summary)
        st.dataframe(cdf, use_container_width=True)

    st.markdown("#### Interactive Confusion Matrix & Performance Report Viewer")
    clf_models = ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree", "svm", "knn"]
    chosen_eval_model = st.selectbox("Select Model to Inspect", clf_models, format_func=lambda x: x.replace("_", " ").title())

    cm_img_path = os.path.join(CLASSIFICATION_DIR, f"{chosen_eval_model}_confusion_matrix.png")
    cm_txt_path = os.path.join(CLASSIFICATION_DIR, f"{chosen_eval_model}_classification_report.txt")

    eval_c1, eval_c2 = st.columns(2)
    with eval_c1:
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption=f"{chosen_eval_model.replace('_', ' ').title()} - Confusion Matrix", use_container_width=True)
        else:
            st.info(f"Confusion matrix plot for {chosen_eval_model} not found.")
    with eval_c2:
        if os.path.exists(cm_txt_path):
            with open(cm_txt_path, "r") as cm_f:
                st.markdown(f"**Classification Report:**")
                st.code(cm_f.read(), language="text")

    st.markdown("---")

    # Section 3: Regression Models (Salary Prediction)
    st.subheader("3. Regression Models for Salary Package")
    reg_summary = os.path.join(REGRESSION_DIR, "regression_model_comparison.csv")
    if os.path.exists(reg_summary):
        rdf = pd.read_csv(reg_summary)
        st.dataframe(rdf, use_container_width=True)

    reg_c1, reg_c2 = st.columns(2)
    with reg_c1:
        act_vs_pred = os.path.join(REGRESSION_DIR, "actual_vs_predicted.png")
        if os.path.exists(act_vs_pred):
            st.image(act_vs_pred, caption="Linear Regression: Actual vs Predicted Salary Package", use_container_width=True)
    with reg_c2:
        res_plot = os.path.join(REGRESSION_DIR, "residual_plot.png")
        if os.path.exists(res_plot):
            st.image(res_plot, caption="Residual Analysis (Homoscedasticity Check)", use_container_width=True)

# =============================================================================
# TAB 8: EDA & GENERATED PLOTS GALLERY
# =============================================================================
with tabs[7]:
    st.header("🖼️ Exploratory & Diagnostic Plots Gallery")
    st.markdown("Browse, filter, and inspect all 190+ generated exploratory data analysis, PCA, clustering, and model diagnostic plots.")

    eda_dir = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "plots")
    eda_out_dir = os.path.join(BASE_DIR, "..", "03_Exploratory_Data_Analysis", "EDA_Outputs")
    pipeline_eda_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "EDA_Outputs")
    hier_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "hierarchical")
    dbscan_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "dbscan")
    pca_dir = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "pca")
    pca_alt_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "pca")
    logistic_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "logistic")
    class_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Classification")
    reg_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Regression")

    all_plots = []
    seen_fpaths = set()

    def register_plot(title, fpath, cat):
        if fpath not in seen_fpaths and os.path.exists(fpath):
            seen_fpaths.add(fpath)
            all_plots.append({"title": title, "path": fpath, "category": cat})

    # 1. PCA Plots
    for pdir in [pca_dir, pca_alt_dir]:
        if os.path.exists(pdir):
            for f in sorted(os.listdir(pdir)):
                if f.endswith(".png"):
                    register_plot(f"[PCA] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(pdir, f), "PCA Projections & Scree")

    # 1b. Anomaly Detection Plots
    anom_dir = os.path.join(BASE_DIR, "..", "04_Models_and_Pipelines", "anomaly_detection")
    anom_alt_dir = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "outputs", "anomaly_detection")
    for adir in [anom_dir, anom_alt_dir]:
        if os.path.exists(adir):
            for f in sorted(os.listdir(adir)):
                if f.endswith(".png"):
                    register_plot(f"[Anomaly] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(adir, f), "Anomaly & Outlier Intelligence")

    # 1c. Data Leakage & Validation Suite Plots (Session 28)
    for ldir in [DATA_LEAKAGE_DIR, DATA_LEAKAGE_OUTPUTS_DIR]:
        if os.path.exists(ldir):
            for f in sorted(os.listdir(ldir)):
                if f.endswith(".png"):
                    register_plot(f"[Data Leakage] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(ldir, f), "Data Leakage & Cross-Validation")

    # 2. DBSCAN Plots
    if os.path.exists(dbscan_dir):
        for f in sorted(os.listdir(dbscan_dir)):
            if f.endswith(".png"):
                register_plot(f"[DBSCAN] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(dbscan_dir, f), "Clustering (DBSCAN & Dendrograms)")

    # 3. Hierarchical Clustering Plots
    if os.path.exists(hier_dir):
        for f in sorted(os.listdir(hier_dir)):
            if f.endswith(".png"):
                register_plot(f"[Hierarchical] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(hier_dir, f), "Clustering (DBSCAN & Dendrograms)")

    # 4. Supervised Classification Confusion Matrices
    if os.path.exists(class_dir):
        for f in sorted(os.listdir(class_dir)):
            if f.endswith("_confusion_matrix.png"):
                register_plot(f"[Confusion Matrix] {f.replace('_confusion_matrix.png', '').replace('_', ' ').title()}", os.path.join(class_dir, f), "Confusion Matrices")

    # 5. Regression Diagnostic Plots
    if os.path.exists(reg_dir):
        for f in sorted(os.listdir(reg_dir)):
            if f.endswith(".png"):
                register_plot(f"[Regression] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(reg_dir, f), "Salary Regression Diagnostics")

    # 6. Logistic Regression Plots
    if os.path.exists(logistic_dir):
        for f in sorted(os.listdir(logistic_dir)):
            if f.endswith(".png"):
                register_plot(f"[Logistic] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(logistic_dir, f), "Confusion Matrices")

    # 7. Overview EDA Outputs
    if os.path.exists(eda_out_dir):
        for f in sorted(os.listdir(eda_out_dir)):
            if f.endswith(".png"):
                register_plot(f"[Overview EDA] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(eda_out_dir, f), "Overview EDA & Heatmaps")

    # 8. Pipeline Preprocessing EDA Outputs
    if os.path.exists(pipeline_eda_dir):
        for f in sorted(os.listdir(pipeline_eda_dir)):
            if f.endswith(".png"):
                register_plot(f"[Pipeline EDA] {f.replace('.png', '').replace('_', ' ').title()}", os.path.join(pipeline_eda_dir, f), "Overview EDA & Heatmaps")

    # 9. Individual Feature EDA Plots
    if os.path.exists(eda_dir):
        for f in sorted(os.listdir(eda_dir)):
            if f.endswith(".png"):
                lower = f.lower()
                if any(k in lower for k in ["cgpa", "sgpa"]):
                    cat = "Academic (CGPA/SGPA)"
                elif any(k in lower for k in ["coding", "aptitude", "mock", "softskill", "workshop", "project", "certification", "publication", "internship"]):
                    cat = "Technical Tests & Skills"
                elif any(k in lower for k in ["gender", "tier", "stream", "city", "hostel", "backlog", "specialisation"]):
                    cat = "Demographics & College Tier"
                elif any(k in lower for k in ["anomaly", "placement"]):
                    cat = "Placement Outcomes & Anomalies"
                else:
                    cat = "General Feature EDA"

                register_plot(f"{f.replace('.png', '').replace('_', ' ').title()}", os.path.join(eda_dir, f), cat)

    categories = [
        "All Categories",
        "Overview EDA & Heatmaps",
        "Academic (CGPA/SGPA)",
        "Technical Tests & Skills",
        "Demographics & College Tier",
        "Placement Outcomes & Anomalies",
        "Clustering (DBSCAN & Dendrograms)",
        "PCA Projections & Scree",
        "Confusion Matrices",
        "Salary Regression Diagnostics",
        "General Feature EDA"
    ]

    col_cat, col_search = st.columns([1, 2])
    with col_cat:
        category_filter = st.selectbox("Filter by Category", categories)
    with col_search:
        search_kw = st.text_input("🔍 Search Plot Name (e.g., Boxplot, Correlation, Attendance, Dendrogram, PCA)", "")

    filtered_items = []
    for p in all_plots:
        if category_filter != "All Categories" and p["category"] != category_filter:
            continue
        if search_kw.strip() and search_kw.lower() not in p["title"].lower():
            continue
        filtered_items.append(p)

    col_info, col_controls = st.columns([2, 1])
    with col_info:
        st.write(f"Displaying **{len(filtered_items)}** plots (out of **{len(all_plots)}** total in project):")
    with col_controls:
        show_all = st.checkbox("Show All Plots (No Page Limit)", value=False)

    if filtered_items:
        page_size = len(filtered_items) if show_all else 24
        total_pages = max(1, (len(filtered_items) + page_size - 1) // page_size)

        if not show_all and total_pages > 1:
            page = st.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_items))
            display_items = filtered_items[start_idx:end_idx]
            st.caption(f"Showing plots {start_idx + 1} to {end_idx} of {len(filtered_items)}")
        else:
            display_items = filtered_items[:page_size]
            st.caption(f"Showing all {len(display_items)} matching plots")

        grid_cols = st.columns(3)
        for idx, item in enumerate(display_items):
            with grid_cols[idx % 3]:
                st.markdown(f"**{item['title']}**")
                st.caption(f"Category: {item['category']}")
                st.image(item["path"], use_container_width=True)
                st.markdown("---")
    else:
        st.info("No plots match the selected category or search filter.")

# =============================================================================
# TAB 9: STUDENT DATABASE EXPLORER
# =============================================================================
with tabs[8]:
    st.header("📋 Student Record Explorer")
    st.markdown("Search, slice, and inspect raw student profiles with real-time statistics.")

    search_id = st.text_input("🔍 Search by Student ID (e.g. 101, 2045)", "")
    
    col_sel, row_limit = st.columns([3, 1])
    with col_sel:
        all_cols = list(df.columns)
        default_cols = [c for c in ["StudentID", "Gender", "Stream", "CollegeTier", "CGPA", "CodingTestScore", "AptitudeTestScore", "AttendancePercent", "PlacementStatusLabel", "Salary Package"] if c in all_cols]
        selected_view_cols = st.multiselect("Visible Columns", all_cols, default=default_cols)
    with row_limit:
        limit = st.number_input("Max Rows Displayed", min_value=10, max_value=2000, value=100, step=50)

    view_df = filtered_df.copy()
    if search_id.strip() != "":
        try:
            val = int(search_id.strip())
            view_df = view_df[view_df["StudentID"] == val]
        except ValueError:
            st.warning("Please enter a valid numeric Student ID.")

    st.markdown(f"Displaying **{min(len(view_df), limit):,}** rows (out of {len(view_df):,} matching):")
    st.dataframe(view_df[selected_view_cols].head(limit), use_container_width=True)

    # Quick summary stats of numeric columns in current view
    if not view_df.empty:
        with st.expander("📊 Summary Statistics of Current View"):
            num_summary = view_df[selected_view_cols].select_dtypes(include=[np.number]).describe()
            st.dataframe(num_summary.round(2), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Placement Analytics & ML Intelligence Suite • Powered by Streamlit, Scikit-Learn & Plotly 🚀")
