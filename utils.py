from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# BASE PATHS
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

from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_models():

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

    product_list = joblib.load(
        os.path.join(
            MODEL_DIR,
            "product_list.pkl"
        )
    )

    df = load_dataset()

    pivot = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(
        pivot.T
    )

    import pandas as pd

    similarity_df = pd.DataFrame(
        similarity,
        index=pivot.columns,
        columns=pivot.columns
    )

    return (
        kmeans,
        scaler,
        similarity_df,
        product_list
    )


# ==========================================================
# CUSTOMER SEGMENTS
# ==========================================================

SEGMENT_MAP = {

    0: "High-Value",

    1: "Regular",

    2: "Occasional",

    3: "At-Risk",

    4: "Emerging",

    5: "Dormant"

}


# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def recommend_product(

    product_name,

    similarity_df,

    top_n=5

):

    product_name = product_name.strip()

    if product_name not in similarity_df.index:

        return None

    recommendations = (

        similarity_df[product_name]

        .sort_values(ascending=False)

        .iloc[1:top_n+1]

    )

    return recommendations.index.tolist()


# ==========================================================
# KPI CALCULATIONS
# ==========================================================

def get_dashboard_metrics(df):

    return {

        "customers": df["CustomerID"].nunique(),

        "products": df["Description"].nunique(),

        "transactions": df["InvoiceNo"].nunique(),

        "revenue": df["TotalAmount"].sum()

    }


# ==========================================================
# MONTHLY SALES
# ==========================================================

def monthly_sales(df):

    monthly = df.copy()

    monthly["Month"] = (

        monthly["InvoiceDate"]

        .dt.to_period("M")

        .astype(str)

    )

    monthly = (

        monthly

        .groupby("Month")["TotalAmount"]

        .sum()

        .reset_index()

    )

    return monthly


# ==========================================================
# COUNTRY SALES
# ==========================================================

def country_sales(df):

    return (

        df.groupby("Country")["TotalAmount"]

        .sum()

        .reset_index()

        .sort_values(

            "TotalAmount",

            ascending=False

        )

    )


# ==========================================================
# TOP PRODUCTS
# ==========================================================

def top_products(df):

    return (

        df.groupby("Description")["Quantity"]

        .sum()

        .reset_index()

        .sort_values(

            "Quantity",

            ascending=False

        )

    )


# ==========================================================
# TOP CUSTOMERS
# ==========================================================

def top_customers(df):

    return (

        df.groupby("CustomerID")["TotalAmount"]

        .sum()

        .reset_index()

        .sort_values(

            "TotalAmount",

            ascending=False

        )

    )
