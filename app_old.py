import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CSS
# ==========================================================

css_path = os.path.join("assets", "style.css")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "cleaned_data",
    "cleaned_online_retail.csv"
)

RFM_PATH = os.path.join(
    BASE_DIR,
    "cleaned_data",
    "rfm_segmented.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@st.cache_data
def load_rfm():
    return pd.read_csv(RFM_PATH)


df = load_data()
rfm_df = load_rfm()

# ==========================================================
# LOAD MODELS
# ==========================================================

kmeans = joblib.load(
    os.path.join(
        MODEL_DIR,
        "kmeans_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

similarity_df = joblib.load(
    os.path.join(
        MODEL_DIR,
        "similarity_matrix.pkl"
    )
)

product_list = joblib.load(
    os.path.join(
        MODEL_DIR,
        "product_list.pkl"
    )
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def recommend_products(product_name, top_n=5):

    product_name = product_name.strip()

    if product_name not in similarity_df.index:
        return None

    rec = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return rec.index.tolist()


segment_map = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk",
    4: "Emerging",
    5: "Dormant"
}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("# 🛒 Shopper Spectrum")

st.sidebar.markdown(
"""
### Customer Analytics Platform

Machine Learning based

✔ Customer Segmentation

✔ Product Recommendation

✔ Business Intelligence
"""
)

st.markdown("""
### Customer Segmentation & Product Recommendation

Analyze customer purchasing behaviour using:

- 📊 RFM Analysis
- 🤖 Machine Learning
- 🎯 Collaborative Filtering
- 📈 Business Intelligence Dashboard
""")


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🛒 Shopper Spectrum")

    st.subheader(
        "Customer Segmentation & Product Recommendation System"
    )

    st.info(
"""
📊 **Dashboard Overview**

This application analyzes customer purchasing behaviour,
segments customers using KMeans clustering,
and recommends similar products using
Item-Based Collaborative Filtering.
"""
)
    
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        df["CustomerID"].nunique()
    )

    c2.metric(
        "Products",
        df["Description"].nunique()
    )

    c3.metric(
        "Transactions",
        df["InvoiceNo"].nunique()
    )

    c4.metric(
        "Revenue",
        f"${df['TotalAmount'].sum():,.2f}"
    )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")
    st.markdown("---")

left, right = st.columns(2)

with left:

    st.subheader("Dataset Summary")

    st.write(f"Rows : {df.shape[0]:,}")

    st.write(f"Columns : {df.shape[1]}")

    st.write(f"Countries : {df['Country'].nunique()}")

with right:

    st.subheader("Project Features")

    st.write("✔ Data Cleaning")

    st.write("✔ EDA")

    st.write("✔ Customer Segmentation")

    st.write("✔ Product Recommendation")

    st.write("✔ Business Insights")

    st.dataframe(
    df.head(15),
    use_container_width=True,
    height=400
)

    st.markdown("---")

    st.subheader("Project Objectives")

    st.markdown("""
- Perform Data Cleaning
- Perform Exploratory Data Analysis (EDA)
- Build RFM Customer Segmentation
- Product Recommendation using Collaborative Filtering
- Deploy an Interactive Streamlit Dashboard
""")

    st.markdown("---")

    st.subheader("📥 Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_online_retail.csv",
        mime="text/csv"
    )

# ==========================================================
# EDA DASHBOARD
# ==========================================================

elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")

    st.write(
        "Analyze customer purchasing behavior, revenue trends, "
        "top-selling products, and customer activity."
    )

    st.markdown("---")

    # ======================================================
    # Country Revenue
    # ======================================================

    country_sales = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .reset_index()
        .sort_values("TotalAmount", ascending=False)
        .head(10)
    )

    fig = px.bar(
        country_sales,
        x="Country",
        y="TotalAmount",
        color="TotalAmount",
        title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Top Products
    # ======================================================

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .reset_index()
        .sort_values("Quantity", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="Quantity",
        y="Description",
        orientation="h",
        color="Quantity",
        title="Top 10 Selling Products"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Monthly Revenue
    # ======================================================

    monthly = df.copy()

    monthly["Month"] = (
        monthly["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sales = (
        monthly.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Month",
        y="TotalAmount",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Revenue Distribution
    # ======================================================

    fig = px.histogram(
        df,
        x="TotalAmount",
        nbins=50,
        title="Revenue Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Top Customers
    # ======================================================

    top_customers = (
        df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .reset_index()
        .sort_values("TotalAmount", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_customers,
        x="CustomerID",
        y="TotalAmount",
        color="TotalAmount",
        title="Top Customers by Spending"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # Correlation Heatmap
    # ======================================================

    st.subheader("Correlation Heatmap")

    corr = df[
        ["Quantity", "UnitPrice", "TotalAmount"]
    ].corr()

    fig, ax = plt.subplots(figsize=(7,5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Business Insights")

    st.success(
        f"💰 Revenue : ${df['TotalAmount'].sum():,.2f}"
    )

    st.info(
        f"👥 Customers : {df['CustomerID'].nunique()}"
    )

    st.info(
        f"📦 Products : {df['Description'].nunique()}"
    )

    st.info(
        f"🧾 Transactions : {df['InvoiceNo'].nunique()}"
    )

# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write(
        "Predict customer category using "
        "Recency, Frequency and Monetary values."
    )

    st.markdown("---")

    # ------------------------------------------------------

    st.subheader("🔍 Search Existing Customer")

    customer_id = st.number_input(
        "Customer ID",
        min_value=0,
        value=0,
        step=1
    )

    if customer_id != 0:

        existing = rfm_df[
            rfm_df["CustomerID"] == customer_id
        ]

        if existing.empty:

            st.warning("Customer not found.")

        else:

            st.success("Customer Found")

            st.dataframe(
                existing,
                use_container_width=True
            )

    st.markdown("---")

    st.subheader("Predict Customer Segment")

    c1, c2, c3 = st.columns(3)

    with c1:

        recency = st.number_input(
            "Recency",
            min_value=0,
            value=30
        )

    with c2:

        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5
        )

    with c3:

        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=500.0
        )

    if st.button("Predict Segment"):

        sample = [[
            recency,
            frequency,
            monetary
        ]]

        sample_scaled = scaler.transform(sample)

        cluster = int(
            kmeans.predict(sample_scaled)[0]
        )

        segment = segment_map.get(
            cluster,
            f"Cluster {cluster}"
        )

        st.success(
            f"Predicted Segment : {segment}"
        )

        if segment == "High-Value":

            st.success("""
High-Value Customers

• Frequent Buyers

• Highest Spending

• Excellent Loyalty

• Reward with Premium Offers
""")

        elif segment == "Regular":

            st.info("""
Regular Customers

• Consistent Purchases

• Medium Spending

• Encourage Repeat Purchases
""")

        elif segment == "Occasional":

            st.warning("""
Occasional Customers

• Purchase Occasionally

• Lower Spending

• Good Target for Promotions
""")

        elif segment == "At-Risk":

            st.error("""
At-Risk Customers

• Long Time Since Last Purchase

• Retention Campaign Recommended
""")

        else:

            st.info(f"Segment : {segment}")

# ==========================================================
# PRODUCT RECOMMENDATION
# ==========================================================

elif page == "🎯 Product Recommendation":
    st.title("🎯 Product Recommendation System")

    st.write(
        "Select a product to get the Top 5 similar product recommendations."
    )

    st.markdown("---")

    product = st.selectbox(
        "Search Product",
        sorted(product_list),
        index=None,
        placeholder="Type or select a product..."
    )

    if st.button("Get Recommendations"):

        if product is None:

            st.warning("Please select a product.")

        else:

            recommendations = recommend_products(product)

            if recommendations is None:

                st.error("Product not found.")

            else:

                st.success(
                    f"Top 5 Recommendations for '{product}'"
                )

                st.markdown("---")

                for i, item in enumerate(recommendations, start=1):

                    with st.container(border=True):

                        st.markdown(
                            f"### 🛍️ {i}. {item}"
                        )

                        st.write(
                            f"Customers who purchased **{product}** also purchased **{item}**."
                        )

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

elif page == "📈 Business Insights":

    st.title("📈 Business Insights")

    st.markdown("---")

    total_revenue = df["TotalAmount"].sum()
    total_customers = df["CustomerID"].nunique()
    total_products = df["Description"].nunique()
    total_transactions = df["InvoiceNo"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"${total_revenue:,.2f}"
    )

    c2.metric(
        "Customers",
        total_customers
    )

    c3.metric(
        "Products",
        total_products
    )

    c4.metric(
        "Transactions",
        total_transactions
    )

    st.markdown("---")

    st.subheader("Customer Segment Distribution")

    segment_counts = (
        rfm_df["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        hole=0.45,
        title="Customer Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Monthly Revenue")

    monthly = df.copy()

    monthly["Month"] = (
        monthly["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sales = (
        monthly.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()
    )

    st.dataframe(
        monthly_sales,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Top Selling Products")

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    st.dataframe(
        top_products,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Download Customer Segmentation")

    csv = rfm_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download RFM Dataset",
        csv,
        "rfm_segmented.csv",
        "text/csv"
    )

    st.markdown("---")

    st.subheader("Business Recommendations")

    st.success("""
✅ Reward High-Value Customers with loyalty programs.
""")

    st.info("""
✅ Encourage Regular Customers with personalized offers.
""")

    st.warning("""
✅ Target Occasional Customers using seasonal campaigns.
""")

    st.error("""
✅ Re-engage At-Risk Customers using discount coupons and email reminders.
""")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Shopper Spectrum")

    st.markdown("""
## 🛒 Shopper Spectrum

An end-to-end Machine Learning and Data Analytics project developed to analyze customer purchasing behaviour in an E-Commerce environment.

### Project Modules

- Data Cleaning
- Exploratory Data Analysis
- RFM Analysis
- Customer Segmentation
- Product Recommendation
- Interactive Streamlit Dashboard

---

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- Joblib
- Streamlit

---

### Machine Learning

- Feature Engineering
- RFM Analysis
- StandardScaler
- KMeans Clustering
- Elbow Method
- Silhouette Score
- Collaborative Filtering
- Cosine Similarity

---

### Business Benefits

- Customer Segmentation
- Product Recommendation
- Sales Analysis
- Business Insights
- Marketing Strategy Support

---

### Developed By

**Varad Kulkarni**
""")
