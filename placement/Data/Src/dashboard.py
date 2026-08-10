import os
import streamlit as st
import pandas as pd
import plotly.express as px

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
    data_path = os.path.join(base_dir, "placement_predict_50k Dataset (2).csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Gender Filter
gender_options = ["All"] + list(df["Gender"].unique())
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Stream Filter
stream_options = ["All"] + list(df["Stream"].unique())
selected_stream = st.sidebar.selectbox("Select Stream", stream_options)

# Placement Status Filter
placement_options = ["All"] + list(df["PlacementStatus"].unique())
selected_placement = st.sidebar.selectbox("Select Placement Status", placement_options)

# Apply filters
filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_stream != "All":
    filtered_df = filtered_df[filtered_df["Stream"] == selected_stream]
if selected_placement != "All":
    filtered_df = filtered_df[filtered_df["PlacementStatus"] == selected_placement]

# -----------------------------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------------------------
st.title("🎓 Placement Prediction Dashboard")
st.markdown("Explore placement statistics, academic performance, and trends interactively.")

# Key Metrics
st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_students = len(filtered_df)
col1.metric("Total Students", f"{total_students:,}")

if total_students > 0:
    placed_count = len(filtered_df[filtered_df["PlacementStatus"] == "Placed"])
    placement_rate = (placed_count / total_students) * 100
    col2.metric("Placement Rate", f"{placement_rate:.1f}%")
    
    avg_cgpa = filtered_df["CGPA"].mean()
    col3.metric("Average CGPA", f"{avg_cgpa:.2f}")

    if "Salary Package" in filtered_df.columns:
        # Assuming salary might have text or numerical format. Convert to numeric if needed.
        # But for safety, we'll try to convert to numeric safely.
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
            color="PlacementStatus",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="CGPA Spread"
        )
        st.plotly_chart(fig_cgpa, use_container_width=True)

    # 2. Placement Status Breakdown
    with row1_col2:
        st.markdown("#### Placement Status Proportion")
        fig_pie = px.pie(
            filtered_df, 
            names='PlacementStatus', 
            title='Placement Status Proportions',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    
    row2_col1, row2_col2 = st.columns(2)

    # 3. CGPA vs Aptitude Test Score
    with row2_col1:
        st.markdown("#### CGPA vs Aptitude Test Score")
        if "AptitudeTestScore" in filtered_df.columns:
            fig_scatter1 = px.scatter(
                filtered_df, 
                x="CGPA", 
                y="AptitudeTestScore", 
                color="PlacementStatus",
                opacity=0.6,
                title="Aptitude vs CGPA"
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)
        else:
            st.info("AptitudeTestScore column not found.")

    # 4. Stream Distribution
    with row2_col2:
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

else:
    st.warning("No data matches the selected filters.")

# Footer
st.markdown("---")
st.caption("Data source: placement_predict_50k Dataset (2).csv | Built with Streamlit")
