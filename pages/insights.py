import streamlit as st
import plotly.express as px

from utils import (
    load_dataset,
    load_rfm,
    get_dashboard_metrics,
    monthly_sales,
    top_products,
    country_sales
)


def show_insights():

    # ======================================================
    # LOAD DATA
    # ======================================================

    df = load_dataset()
    rfm_df = load_rfm()

    metrics = get_dashboard_metrics(df)

    st.title("📈 Business Insights Dashboard")

    st.markdown("""
Executive summary of customer behaviour, sales performance,
and business recommendations.
""")

    st.markdown("---")

    # ======================================================
    # KPI CARDS
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Revenue",
        f"${metrics['revenue']:,.2f}"
    )

    c2.metric(
        "👥 Customers",
        metrics["customers"]
    )

    c3.metric(
        "📦 Products",
        metrics["products"]
    )

    c4.metric(
        "🧾 Transactions",
        metrics["transactions"]
    )

    st.markdown("---")

    # ======================================================
    # HIGHEST REVENUE COUNTRY
    # ======================================================

    st.subheader("🌍 Top Revenue Countries")

    country_df = country_sales(df).head(10)

    fig = px.bar(
        country_df,
        x="Country",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # CUSTOMER SEGMENT DISTRIBUTION
    # ======================================================

    st.subheader("👥 Customer Segment Distribution")

    segment_df = (
        rfm_df["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_df.columns = [
        "Segment",
        "Customers"
    ]

    fig = px.pie(
        segment_df,
        names="Segment",
        values="Customers",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # MONTHLY SALES
    # ======================================================

    st.subheader("📅 Monthly Revenue")

    monthly_df = monthly_sales(df)

    fig = px.line(
        monthly_df,
        x="Month",
        y="TotalAmount",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        monthly_df,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # TOP PRODUCTS
    # ======================================================

    st.subheader("🏆 Top Selling Products")

    products = top_products(df).head(20)

    st.dataframe(
        products,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # DOWNLOAD RFM
    # ======================================================

    st.subheader("📥 Download Customer Segmentation")

    csv = rfm_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download RFM Dataset",
        csv,
        "rfm_segmented.csv",
        "text/csv"
    )

    st.markdown("---")

    # ======================================================
    # BUSINESS RECOMMENDATIONS
    # ======================================================

    st.subheader("💡 Business Recommendations")

    st.success("""
### 🌟 High-Value Customers

Reward with loyalty programs, exclusive offers,
and premium memberships.
""")

    st.info("""
### 👍 Regular Customers

Increase repeat purchases using reward points
and personalized discounts.
""")

    st.warning("""
### 🛍️ Occasional Customers

Target with seasonal campaigns,
coupon codes,
and product bundles.
""")

    st.error("""
### ⚠️ At-Risk Customers

Launch retention campaigns,
email reminders,
and special discounts.
""")

    st.markdown("---")

    st.subheader("📌 Overall Conclusion")

    st.info("""
Customer Segmentation combined with Product Recommendation
helps businesses improve customer retention,
increase revenue,
optimize inventory,
and deliver personalized shopping experiences.
""")
    