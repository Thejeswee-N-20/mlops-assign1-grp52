#Unit tests for data loading functionality.

import os
from src.data_loader import load_data


def test_load_data_returns_dataframe():
    # Test that load_data successfully loads the dataset.

    df = load_data("data/processed/heart_cleaned.csv")

    # Basic sanity checks
    assert df is not None
    assert not df.empty
    assert "target" in df.columns
