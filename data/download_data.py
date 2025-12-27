"""
Dataset Source:
UCI Machine Learning Repository - Heart Disease Dataset
https://archive.ics.uci.edu/dataset/45/heart+disease

This script downloads the Cleveland subset of the Heart Disease dataset
and converts it into a clean CSV suitable for ML pipelines.
"""

import os
import pandas as pd
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

RAW_DATA_PATH = "data/raw/processed.cleveland.data"
PROCESSED_DATA_PATH = "data/processed/heart_cleaned.csv"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def download_data():
    os.makedirs("data/raw", exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
    print(f"Raw data downloaded to {RAW_DATA_PATH}")


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES)

    # Replace '?' with NaN and drop missing values
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert columns to numeric
    df = df.apply(pd.to_numeric)

    # Convert target to binary (0 = no disease, 1 = disease)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    download_data()
    clean_data()