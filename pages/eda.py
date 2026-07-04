import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_dataset,
    monthly_sales,
    country_sales,
    top_products,
    top_customers
)


def show_eda():

    df = load_dataset()

    st.title("📊 Exploratory Data Analysis Dashboard")

    st.markdown("""
Analyze customer purchasing behaviour through interactive visualizations.
""")

    st.markdown("---")

    # =====================================================
    # FILTER
    # =====================================================

    countries = ["All"] + sorted(df["Country"].unique().tolist())

    selected_country = st.selectbox(
        "🌍 Filter by Country",
        countries
    )

    if selected_country != "All":
        filtered_df = df[df["Country"] == selected_country]
    else:
        filtered_df = df.copy()

    st.markdown("---")

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"${filtered_df['TotalAmount'].sum():,.0f}"
    )

    c2.metric(
        "Customers",
        filtered_df["CustomerID"].nunique()
    )

    c3.metric(
        "Products",
        filtered_df["Description"].nunique()
    )

    c4.metric(
        "Transactions",
        filtered_df["InvoiceNo"].nunique()
    )

    st.markdown("---")

    # =====================================================
    # COUNTRY REVENUE
    # =====================================================

    st.subheader("🌍 Top Countries by Revenue")

    country_df = country_sales(filtered_df).head(10)

    fig = px.bar(
        country_df,
        x="Country",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Country",
        yaxis_title="Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TOP PRODUCTS
    # =====================================================

    st.subheader("🏆 Top Selling Products")

    product_df = top_products(filtered_df).head(10)

    fig = px.bar(
        product_df,
        x="Quantity",
        y="Description",
        orientation="h",
        color="Quantity",
        text_auto=True
    )

    fig.update_layout(height=550)

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # MONTHLY SALES
    # =====================================================

    st.subheader("📈 Monthly Revenue Trend")

    monthly_df = monthly_sales(filtered_df)

    fig = px.line(
        monthly_df,
        x="Month",
        y="TotalAmount",
        markers=True
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # REVENUE DISTRIBUTION
    # =====================================================

    st.subheader("💰 Revenue Distribution")

    fig = px.histogram(
        filtered_df,
        x="TotalAmount",
        nbins=50
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # TOP CUSTOMERS
    # =====================================================

    st.subheader("👥 Top Customers")

    customer_df = top_customers(filtered_df).head(10)

    fig = px.bar(
        customer_df,
        x="CustomerID",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))

    corr = filtered_df[
        [
            "Quantity",
            "UnitPrice",
            "TotalAmount"
        ]
    ].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    st.subheader("📌 Business Insights")

    c1, c2 = st.columns(2)

    with c1:

        st.success(
            f"""
Revenue Generated

${filtered_df['TotalAmount'].sum():,.2f}
"""
        )

        st.info(
            f"""
Unique Customers

{filtered_df['CustomerID'].nunique()}
"""
        )

    with c2:

        st.info(
            f"""
Products

{filtered_df['Description'].nunique()}
"""
        )

        st.info(
            f"""
Transactions

{filtered_df['InvoiceNo'].nunique()}
"""
        )

    st.markdown("---")

    st.subheader("📋 Key Observations")

    st.markdown("""
- Revenue is concentrated in a few countries.

- A small number of products generate most sales.

- Customer purchasing behaviour varies significantly.

- Revenue trends indicate seasonal purchasing patterns.

- Insights from this dashboard support inventory planning,
marketing campaigns and customer retention strategies.
""")
    