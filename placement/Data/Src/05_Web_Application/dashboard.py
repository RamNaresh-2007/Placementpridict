import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Placement Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "01_Datasets", "placement_predict_50k Dataset (2).csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    if "PlacementStatus" in df.columns:
        df["PlacementStatusLabel"] = df["PlacementStatus"].map({1: "Placed", 0: "Not Placed"}).fillna(df["PlacementStatus"])
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Filters")

# Gender Filter
gender_options = ["All"] + list(df["Gender"].unique())
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Stream Filter
stream_options = ["All"] + list(df["Stream"].unique())
selected_stream = st.sidebar.selectbox("Select Stream", stream_options)

# Placement Status Filter
placement_options = ["All"] + list(df["PlacementStatusLabel"].unique())
selected_placement = st.sidebar.selectbox("Select Placement Status", placement_options)

# Apply filters
filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_stream != "All":
    filtered_df = filtered_df[filtered_df["Stream"] == selected_stream]
if selected_placement != "All":
    filtered_df = filtered_df[filtered_df["PlacementStatusLabel"] == selected_placement]

# -----------------------------------------------------------------------------
# Main Dashboard Header
# -----------------------------------------------------------------------------
st.title("🎓 Placement Prediction Dashboard")
st.markdown("Explore placement statistics, run what-if analyses, and review learning models interactively.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Analytics Overview", "🧠 What-If Predictor", "📈 Models & Pipeline"])

# =============================================================================
# TAB 1: ANALYTICS OVERVIEW
# =============================================================================
with tab1:
    # Key Metrics
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)

    total_students = len(filtered_df)
    col1.metric("Total Students", f"{total_students:,}")

    if total_students > 0:
        placed_count = len(filtered_df[filtered_df["PlacementStatusLabel"] == "Placed"])
        placement_rate = (placed_count / total_students) * 100
        col2.metric("Placement Rate", f"{placement_rate:.1f}%")
        
        avg_cgpa = filtered_df["CGPA"].mean()
        col3.metric("Average CGPA", f"{avg_cgpa:.2f}")

        if "Salary Package" in filtered_df.columns:
            salary_numeric = pd.to_numeric(filtered_df["Salary Package"], errors='coerce')
            avg_salary = salary_numeric.mean()
            if pd.notna(avg_salary):
                col4.metric("Avg Salary Package", f"{avg_salary/100000:.2f} LPA")
            else:
                col4.metric("Avg Salary Package", "N/A")
    else:
        col2.metric("Placement Rate", "N/A")
        col3.metric("Average CGPA", "N/A")
        col4.metric("Avg Salary Package", "N/A")

    st.markdown("---")

    # Visualizations
    if total_students > 0:
        st.subheader("Visualizations")
        
        row1_col1, row1_col2 = st.columns(2)

        # 1. CGPA Distribution
        with row1_col1:
            st.markdown("#### CGPA Distribution")
            fig_cgpa = px.histogram(
                filtered_df, 
                x="CGPA", 
                nbins=30, 
                color="PlacementStatusLabel",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="CGPA Spread"
            )
            st.plotly_chart(fig_cgpa, use_container_width=True)

        # 2. Placement Status Breakdown
        with row1_col2:
            st.markdown("#### Placement Status Proportion")
            fig_pie = px.pie(
                filtered_df, 
                names='PlacementStatusLabel', 
                title='Placement Status Proportions',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        
        # 3. New Feature: Correlation Heatmap (Numeric only)
        st.markdown("#### 🌟 Feature Correlation Heatmap")
        numeric_df = filtered_df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig_corr = px.imshow(corr_matrix, 
                                 text_auto=".2f", 
                                 aspect="auto",
                                 color_continuous_scale="RdBu_r",
                                 title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        
        row2_col1, row2_col2 = st.columns(2)

        # 4. Stream Distribution
        with row2_col1:
            st.markdown("#### Student Distribution by Stream")
            if "Stream" in filtered_df.columns:
                stream_counts = filtered_df["Stream"].value_counts().reset_index()
                stream_counts.columns = ["Stream", "Count"]
                fig_bar = px.bar(
                    stream_counts, 
                    x="Stream", 
                    y="Count", 
                    color="Stream",
                    title="Students per Stream"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Stream column not found.")

        # 5. New Feature: 3D Scatter Plot (CGPA vs Coding vs Aptitude)
        with row2_col2:
            st.markdown("#### 🌟 3D Placement Landscape")
            if all(col in filtered_df.columns for col in ["CGPA", "CodingTestScore", "AptitudeTestScore"]):
                fig_3d = px.scatter_3d(
                    filtered_df.head(1000), # Sample to keep it performant
                    x='CGPA', y='CodingTestScore', z='AptitudeTestScore',
                    color='PlacementStatusLabel',
                    opacity=0.7,
                    title='CGPA vs Coding vs Aptitude (Sampled)'
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Missing columns for 3D plot.")

        # Download Data Feature
        st.markdown("---")
        st.markdown("#### 💾 Export Filtered Data")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='filtered_placement_data.csv',
            mime='text/csv',
        )

    else:
        st.warning("No data matches the selected filters.")

# =============================================================================
# TAB 2: WHAT-IF PREDICTOR
# =============================================================================
with tab2:
    st.header("🧠 AI-Powered What-If Predictor")
    st.markdown("Using a real, trained **Logistic Regression Pipeline**, adjust your profile below to get live AI predictions for your placement chances!")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("Your Profile")
        sim_cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
        sim_aptitude = st.slider("Aptitude Test Score (%)", 0, 100, 70, 1)
        sim_coding = st.slider("Coding Test Score (%)", 0, 100, 65, 1)
        sim_mock = st.slider("Mock Interview Score (%)", 0, 100, 75, 1)
        sim_attendance = st.slider("Attendance (%)", 0, 100, 85, 1)
        
        with st.expander("Advanced Options"):
            sim_internships = st.number_input("Internships", min_value=0, max_value=5, value=1)
            sim_projects = st.number_input("Projects", min_value=0, max_value=10, value=2)
            sim_certifications = st.number_input("Certifications", min_value=0, max_value=10, value=1)
            sim_gender = st.selectbox("Gender", ["Male", "Female"])
            sim_stream = st.selectbox("Stream", ["CS", "IT", "ECE", "Mechanical", "EE", "Civil"])
    
    with col_output:
        st.subheader("AI Prediction Output")
        
        # Load the real AI Pipeline
        import joblib
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs", "Model_Outputs", "Classification", "logistic_regression.pkl")
        
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            
            # Construct a DataFrame matching the training data structure
            input_dict = {
                'SGPA_Sem1': [sim_cgpa], 'SGPA_Sem2': [sim_cgpa], 'SGPA_Sem3': [sim_cgpa], 'SGPA_Sem4': [sim_cgpa],
                'SGPA_Sem5': [sim_cgpa], 'SGPA_Sem6': [sim_cgpa], 'SGPA_Sem7': [sim_cgpa], 'SGPA_Sem8': [sim_cgpa],
                'CGPA': [sim_cgpa], 'AttendancePercent': [sim_attendance],
                'Internships': [sim_internships], 'Projects': [sim_projects], 
                'Workshops': [0], 'Certifications': [sim_certifications], 'Publications': [0],
                'AptitudeTestScore': [sim_aptitude], 'SoftSkillsRating': [80.0],
                'CodingTestScore': [sim_coding], 'MockInterviewScore': [sim_mock], 'ExtraCurricular': [1],
                'Gender': [sim_gender], 'City': ['Pune'], 'CollegeTier': ['Tier2'],
                'Stream': [sim_stream], 'Specialisation': ['AI'], 'Hostel': ['No'], 'HistoryOfBacklogs': ['No']
            }
            input_df = pd.DataFrame(input_dict)
            
            # Real Prediction!
            prob = pipeline.predict_proba(input_df)[0][1]
            prob_percent = prob * 100
            
            st.markdown(f"### AI Estimated Placement Probability: **{prob_percent:.1f}%**")
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "khaki"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prob > 0.7:
                st.success("🎉 High chances of placement! Excellent profile.")
            elif prob > 0.4:
                st.warning("⚠️ Borderline chances. Focus on improving your Coding and Mock Interview scores.")
            else:
                st.error("📉 Low chances. Consider significant improvements in CGPA and technical skills.")
                
            # Smart AI Insight Generator
            st.markdown("--- ")
            st.markdown("#### 🤖 AI Profile Analyzer")
            
            # Test what a +15% coding score would do
            input_dict_better = input_dict.copy()
            input_dict_better['CodingTestScore'] = [min(100.0, sim_coding + 15)]
            prob_better = pipeline.predict_proba(pd.DataFrame(input_dict_better))[0][1]
            
            if (prob_better - prob) > 0.05:
                st.info(f"💡 **AI Insight**: If you increase your Coding Score by 15%, your placement chance jumps to **{prob_better*100:.1f}%**!")
            else:
                st.info("💡 **AI Insight**: Your coding score is solid. Try doing more projects or securing an internship to stand out!")
                
        else:
            st.error("Real ML Pipeline model not found! Please run the complete pipeline script first to generate `logistic_regression.pkl`.")

# =============================================================================
# TAB 3: MODELS & PIPELINE
# =============================================================================
with tab3:
    st.header("🤖 Models & Scaling Comparison")
    st.markdown("This section demonstrates the impact of feature scaling on different families of machine learning models.")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Outputs", "model_scaling_results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        
        st.subheader("Model Performance Results")
        st.dataframe(results_df, use_container_width=True)
        
        st.subheader("Visualizing Scaling Impact")
        fig_metrics = px.bar(
            results_df,
            x="Model Name",
            y="Score",
            color="Scaled",
            barmode="group",
            facet_col="Task",
            title="Metric Scores: Scaled vs Unscaled (Higher is Better)"
        )
        fig_metrics.update_yaxes(matches=None)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        fig_time = px.bar(
            results_df,
            x="Model Name",
            y="Train Time (s)",
            color="Scaled",
            barmode="group",
            facet_col="Task",
            title="Training Time (s): Scaled vs Unscaled (Lower is generally better)"
        )
        fig_time.update_yaxes(matches=None)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Model results not found. Run `python Models.py` to generate the comparison.")

    st.markdown("---")
    st.header("🚀 Comprehensive Pipeline Results")
    st.markdown("Results from the latest end-to-end pipeline (`complete_pipeline.py`).")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_out = os.path.join(base_dir, "..", "06_Outputs_and_Utils", "PlacementPredict_Outputs")

    summary_path = os.path.join(pipeline_out, "complete_project_summary.csv")
    if os.path.exists(summary_path):
        st.subheader("Executive Summary")
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df.T, use_container_width=True)

    col_class, col_reg = st.columns(2)

    class_path = os.path.join(pipeline_out, "Model_Outputs", "Classification", "classification_model_comparison.csv")
    if os.path.exists(class_path):
        with col_class:
            st.subheader("Classification Models")
            class_df = pd.read_csv(class_path)
            st.dataframe(class_df, use_container_width=True)
            if "F1" in class_df.columns:
                fig_clf = px.bar(class_df, x="Model", y="F1", color="Model", title="F1 Score Comparison")
                st.plotly_chart(fig_clf, use_container_width=True)

    reg_path = os.path.join(pipeline_out, "Model_Outputs", "Regression", "regression_model_comparison.csv")
    if os.path.exists(reg_path):
        with col_reg:
            st.subheader("Regression Models")
            reg_df = pd.read_csv(reg_path)
            st.dataframe(reg_df, use_container_width=True)
            if "R2" in reg_df.columns:
                fig_reg = px.bar(reg_df, x="Model", y="R2", color="Model", title="R-Squared (R2) Comparison")
                st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")
    st.header("📚 Newly Covered Topics")
    st.markdown("""
    The following topics have been integrated into the learning pipelines and models:
    - **Decision Trees**: Classifiers and Regressors, pruning techniques, Information Gain, and Gini Impurity.
    - **Linear & Logistic Regression**: Normal equations vs Gradient descent, Sigmoid functions, and Cross-entropy loss.
    - **Regularization**: Ridge (L2), Lasso (L1), and Elastic Net preventing overfitting.
    - **Advanced Logistic Regression**: Multinomial Logistic Regression (Softmax Regression) for multi-class targets.
    - **Feature Encoding**: One-Hot, Ordinal, and Target Encoding for categorical variables.
    - **Imputation & Scaling**: Missing values handling strategies, `StandardScaler`, and `MinMaxScaler` implications on distance-based vs tree-based models.
    """)

# Footer
st.markdown("---")
st.caption("Data source: placement_predict_50k Dataset (2).csv | Built with Streamlit | Updated with Interactive Features 🚀")
