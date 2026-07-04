import streamlit as st
from utils import load_dataset, get_dashboard_metrics


def show_home():

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    df = load_dataset()

    metrics = get_dashboard_metrics(df)

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    st.title("🛒 Shopper Spectrum")

    st.markdown("""
### Customer Segmentation & Product Recommendation System

Analyze customer purchasing behaviour using:

- 📊 RFM Analysis
- 🤖 Machine Learning
- 🎯 Collaborative Filtering
- 📈 Business Intelligence Dashboard
""")

    st.info("""
This dashboard helps analyze customer purchasing behaviour,
identify valuable customer segments,
and recommend products using Machine Learning.
""")

    st.markdown("---")

    # --------------------------------------------------
    # KPI Cards
    # --------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Customers",
        f"{metrics['customers']:,}"
    )

    c2.metric(
        "📦 Products",
        f"{metrics['products']:,}"
    )

    c3.metric(
        "🧾 Transactions",
        f"{metrics['transactions']:,}"
    )

    c4.metric(
        "💰 Revenue",
        f"${metrics['revenue']:,.2f}"
    )

    st.markdown("---")

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True,
        height=420
    )

    st.markdown("---")

    # --------------------------------------------------
    # Download Dataset
    # --------------------------------------------------

    st.subheader("📥 Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="cleaned_online_retail.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # --------------------------------------------------
    # Project Overview
    # --------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("📌 Project Objectives")

        st.markdown("""
- Data Cleaning
- Exploratory Data Analysis
- RFM Analysis
- Customer Segmentation
- Product Recommendation
- Business Intelligence Dashboard
""")

    with right:

        st.subheader("⚙️ Technologies Used")

        st.markdown("""
- Python
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Joblib
""")

    st.markdown("---")

    # --------------------------------------------------
    # Quick Statistics
    # --------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📊 Dataset Summary")

        st.write(f"Rows : {df.shape[0]:,}")
        st.write(f"Columns : {df.shape[1]}")
        st.write(f"Countries : {df['Country'].nunique()}")

    with c2:

        st.subheader("✨ Features")

        st.write("✔ Customer Segmentation")
        st.write("✔ Product Recommendation")
        st.write("✔ Interactive Dashboard")
        st.write("✔ Download Reports")
        st.write("✔ Business Insights")
        