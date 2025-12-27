# defines feature preprocessing steps such as scaling numerical features and preparing data for modeling.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_preprocessing_pipeline(numerical_features):
    """
    Create a preprocessing pipeline for numerical features.

    Parameters:
        numerical_features (list): List of numerical column names

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """

    # Scale numerical features to bring them to a similar range
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features)
        ]
    )

    return preprocessor
