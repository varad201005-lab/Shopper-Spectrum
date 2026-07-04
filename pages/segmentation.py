import streamlit as st
import pandas as pd

from utils import (
    load_rfm,
    load_models,
    SEGMENT_MAP
)


def show_segmentation():

    st.title("👥 Customer Segmentation")

    st.markdown("""
Analyze customer groups using **Recency, Frequency, and Monetary (RFM)** values.
""")

    st.markdown("---")

    # -------------------------------------------------------
    # Load Data and Models
    # -------------------------------------------------------

    rfm_df = load_rfm()

    kmeans, scaler, _, _ = load_models()

    # -------------------------------------------------------
    # Existing Customer Search
    # -------------------------------------------------------

    st.subheader("🔍 Search Existing Customer")

    customer_ids = sorted(rfm_df["CustomerID"].unique())

    customer = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    if st.button("View Customer Details"):

        details = rfm_df[rfm_df["CustomerID"] == customer]

        st.success("Customer Found")

        st.dataframe(
            details,
            use_container_width=True
        )

    st.markdown("---")

    # -------------------------------------------------------
    # Predict Customer Segment
    # -------------------------------------------------------

    st.subheader("🎯 Predict Customer Segment")

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

        sample = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        sample_scaled = scaler.transform(sample)

        cluster = int(kmeans.predict(sample_scaled)[0])

        segment = SEGMENT_MAP.get(
            cluster,
            f"Cluster {cluster}"
        )

        st.markdown("---")

        st.success(f"### Predicted Segment: {segment}")

        # ---------------------------------------------------
        # Segment Description
        # ---------------------------------------------------

        if segment == "High-Value":

            st.info("""
### 🌟 High-Value Customer

- Frequent Purchases
- High Spending
- Loyal Customer
- Ideal for Premium Memberships
- Reward with Exclusive Offers
""")

        elif segment == "Regular":

            st.info("""
### 👍 Regular Customer

- Consistent Purchases
- Medium Spending
- Responds Well to Loyalty Programs
- Encourage Repeat Purchases
""")

        elif segment == "Occasional":

            st.warning("""
### 🛍️ Occasional Customer

- Purchases Occasionally
- Lower Spending
- Target with Seasonal Promotions
""")

        elif segment == "At-Risk":

            st.error("""
### ⚠️ At-Risk Customer

- Long Time Since Last Purchase
- Very Low Activity
- Needs Retention Campaigns
""")

        elif segment == "Emerging":

            st.success("""
### 🚀 Emerging Customer

- New Customer
- Growth Potential
- Encourage Repeat Purchases
""")

        elif segment == "Dormant":

            st.error("""
### ❌ Dormant Customer

- No Recent Purchases
- Lowest Engagement
- Win-back Campaign Required
""")

    st.markdown("---")

    # -------------------------------------------------------
    # Segment Distribution
    # -------------------------------------------------------

    st.subheader("📊 Customer Segment Distribution")

    segment_counts = (
        rfm_df["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    st.dataframe(
        segment_counts,
        use_container_width=True
    )
    