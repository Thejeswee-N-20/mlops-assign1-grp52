# Unit tests for model training pipeline.


from src.data_loader import load_data
from src.features import create_preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_model_pipeline_runs():
    # Test that the preprocessing + model pipeline can be fit without errors.
    df = load_data("data/processed/heart_cleaned.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    numerical_features = X.columns.tolist()
    preprocessor = create_preprocessing_pipeline(numerical_features)

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Fit the pipeline
    pipeline.fit(X, y)

    assert pipeline is not None
