import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ==========================================================
# SHOPPER SPECTRUM
# CUSTOMER SEGMENTATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "cleaned_data",
    "cleaned_online_retail.csv"
)

MODEL_PATH = os.path.join(BASE_DIR, "models")
IMAGE_PATH = os.path.join(BASE_DIR, "images")
OUTPUT_PATH = os.path.join(BASE_DIR, "cleaned_data")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)

# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Dataset Loaded Successfully")

# ==========================================================
# RFM FEATURE ENGINEERING
# ==========================================================

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({

    "InvoiceDate": lambda x: (reference_date - x.max()).days,

    "InvoiceNo": "nunique",

    "TotalAmount": "sum"

})

rfm.columns = [

    "Recency",

    "Frequency",

    "Monetary"

]

print("\nRFM Dataset Created")

print(rfm.head())

# ==========================================================
# SAVE RFM DATASET
# ==========================================================

rfm.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "rfm_data.csv"
    )
)

# ==========================================================
# STANDARDIZATION
# ==========================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(rfm)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_PATH,
        "scaler.pkl"
    )
)

# ==========================================================
# ELBOW METHOD
# ==========================================================

wcss = []

for i in range(2,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(2,11),wcss,marker="o")

plt.title("Elbow Method")

plt.xlabel("Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig(
    os.path.join(
        IMAGE_PATH,
        "elbow_method.png"
    )
)

plt.close()

# ==========================================================
# SILHOUETTE SCORE
# ==========================================================

scores = {}

for i in range(2,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    score = silhouette_score(
        scaled_data,
        labels
    )

    scores[i] = score

best_cluster = max(
    scores,
    key=scores.get
)

print("\nSilhouette Scores")

for k,v in scores.items():

    print(k,"Clusters :",round(v,4))

print("\nBest Cluster :",best_cluster)

# ==========================================================
# FINAL MODEL
# ==========================================================

kmeans = KMeans(

    n_clusters=best_cluster,

    random_state=42,

    n_init=10

)

rfm["Cluster"] = kmeans.fit_predict(
    scaled_data
)

joblib.dump(

    kmeans,

    os.path.join(

        MODEL_PATH,

        "kmeans_model.pkl"

    )

)

# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

summary = rfm.groupby("Cluster")[

    ["Recency","Frequency","Monetary"]

].mean()

print("\nCluster Summary\n")

print(summary)

# ==========================================================
# LABEL CLUSTERS
# ==========================================================

summary = summary.sort_values(
    by="Monetary",
    ascending=False
)

labels = {}

segment_names = [

    "High-Value",

    "Regular",

    "Occasional",

    "At-Risk",

    "Emerging",

    "Dormant"

]

for cluster,segment in zip(

    summary.index,

    segment_names

):

    labels[cluster]=segment

rfm["Segment"] = rfm["Cluster"].map(labels)

# ==========================================================
# SAVE UPDATED RFM
# ==========================================================

rfm.to_csv(

    os.path.join(

        OUTPUT_PATH,

        "rfm_segmented.csv"

    )

)

# ==========================================================
# SCATTER PLOT
# ==========================================================

plt.figure(figsize=(9,6))

sns.scatterplot(

    data=rfm,

    x="Frequency",

    y="Monetary",

    hue="Segment",

    palette="Set2",

    s=70

)

plt.title("Customer Segments")

plt.savefig(

    os.path.join(

        IMAGE_PATH,

        "customer_segments.png"

    )

)

plt.close()

print("\nModel Saved Successfully")

print("\nCustomer Segmentation Completed")
