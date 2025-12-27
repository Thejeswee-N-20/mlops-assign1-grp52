# load the processed heart disease dataset, Separating data loading logic makes the pipeline cleaner and reusable.

import os
import pandas as pd


def load_data(file_path: str):
    """
    Load the cleaned heart disease dataset from CSV.

    Parameters:
        file_path (str): Relative path from project root

    Returns:
        pd.DataFrame: Loaded dataset
    """

    # Get absolute path of the current file (data_loader.py)
    current_dir = os.path.dirname(__file__)

    # Move one level up to project root
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # Construct full path to the dataset
    full_path = os.path.join(project_root, file_path)

    df = pd.read_csv(full_path)
    return df
