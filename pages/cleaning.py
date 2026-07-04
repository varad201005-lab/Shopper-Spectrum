import os
import pandas as pd

# ==========================================================
# SHOPPER SPECTRUM
# DATA CLEANING
# ==========================================================

# Project folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "online_retail.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "cleaned_data")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("SHOPPER SPECTRUM - DATA CLEANING")
print("=" * 60)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="ISO-8859-1"
)

print("Dataset Loaded Successfully\n")

# ----------------------------------------------------------
# Basic Information
# ----------------------------------------------------------

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nDataset Shape :", df.shape)

print("\nColumns")

print(df.columns.tolist())

# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

print("\nChecking Missing Values...\n")

print(df.isnull().sum())

# ----------------------------------------------------------
# Duplicate Records
# ----------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

# ----------------------------------------------------------
# Remove Missing CustomerID
# ----------------------------------------------------------

rows_before = len(df)

df = df.dropna(subset=["CustomerID"])

rows_after = len(df)

print("\nRemoved Missing CustomerID :", rows_before - rows_after)

# ----------------------------------------------------------
# Remove Cancelled Orders
# ----------------------------------------------------------

rows_before = len(df)

df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
]

rows_after = len(df)

print("Cancelled Orders Removed :", rows_before - rows_after)

# ----------------------------------------------------------
# Remove Invalid Quantity
# ----------------------------------------------------------

rows_before = len(df)

df = df[df["Quantity"] > 0]

rows_after = len(df)

print("Negative/Zero Quantity Removed :", rows_before - rows_after)

# ----------------------------------------------------------
# Remove Invalid UnitPrice
# ----------------------------------------------------------

rows_before = len(df)

df = df[df["UnitPrice"] > 0]

rows_after = len(df)

print("Negative/Zero Price Removed :", rows_before - rows_after)

# ----------------------------------------------------------
# Remove Duplicate Records
# ----------------------------------------------------------

rows_before = len(df)

df = df.drop_duplicates()

rows_after = len(df)

print("Duplicate Rows Removed :", rows_before - rows_after)

# ----------------------------------------------------------
# Convert Data Types
# ----------------------------------------------------------

print("\nConverting Data Types...")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["CustomerID"] = df["CustomerID"].astype(int)

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------

print("\nCreating TotalAmount Column...")

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# ----------------------------------------------------------
# Final Dataset Information
# ----------------------------------------------------------

print("\nFinal Dataset Shape :", df.shape)

print("\nPreview")

print(df.head())

# ----------------------------------------------------------
# Save Clean Dataset
# ----------------------------------------------------------

clean_path = os.path.join(
    OUTPUT_FOLDER,
    "cleaned_online_retail.csv"
)

df.to_csv(
    clean_path,
    index=False
)

print("\nClean Dataset Saved Successfully")

print(clean_path)

print("\nCleaning Completed Successfully!")
