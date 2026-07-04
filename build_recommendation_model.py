import os
import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "cleaned_data",
    "cleaned_online_retail.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# ==========================================================
# KEEP ONLY VALID RECORDS
# ==========================================================

df = df[df["CustomerID"].notna()]
df = df[df["Quantity"] > 0]

# ==========================================================
# KEEP TOP 500 PRODUCTS
# ==========================================================

top_products = (
    df.groupby("Description")["Quantity"]
      .sum()
      .sort_values(ascending=False)
      .head(500)
      .index
)

df = df[df["Description"].isin(top_products)]

print(f"Products retained: {len(top_products)}")

# ==========================================================
# CUSTOMER-PRODUCT MATRIX
# ==========================================================

pivot = pd.pivot_table(
    df,
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("Pivot Shape:", pivot.shape)

# ==========================================================
# COSINE SIMILARITY
# ==========================================================

similarity = cosine_similarity(pivot.T)

similarity_df = pd.DataFrame(
    similarity,
    index=pivot.columns,
    columns=pivot.columns
)

# ==========================================================
# SAVE FILES
# ==========================================================

joblib.dump(
    similarity_df,
    os.path.join(
        MODEL_PATH,
        "similarity_matrix.pkl"
    ),
    compress=3
)

joblib.dump(
    list(similarity_df.index),
    os.path.join(
        MODEL_PATH,
        "product_list.pkl"
    )
)

print("\nRecommendation model created successfully!")

size = os.path.getsize(
    os.path.join(
        MODEL_PATH,
        "similarity_matrix.pkl"
    )
) / (1024 * 1024)

print(f"Similarity Matrix Size : {size:.2f} MB")
