# trains multiple classification models on the heart disease dataset and evaluates them using cross-validation.

import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from src.data_loader import load_data
from src.features import create_preprocessing_pipeline


def train_and_evaluate():
    
    # Set MLflow experiment
    mlflow.set_experiment("Heart Disease Classification")

    # Load processed dataset
    df = load_data("data/processed/heart_cleaned.csv")

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # All features are numerical
    numerical_features = X.columns.tolist()

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_features)

    # Define models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # Evaluation metrics
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining and evaluating: {model_name}")

        with mlflow.start_run(run_name=model_name):

            # Log model-specific parameters
            if model_name == "Logistic Regression":
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", model.max_iter)

            if model_name == "Random Forest":
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("random_state", model.random_state)

            # Create full pipeline (preprocessing + model)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            # Perform 5-fold cross-validation
            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=5,
                scoring=scoring,
                return_estimator=True
            )

            # Calculate average metrics
            accuracy = cv_results["test_accuracy"].mean()
            precision = cv_results["test_precision"].mean()
            recall = cv_results["test_recall"].mean()
            roc_auc = cv_results["test_roc_auc"].mean()

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            # Print results to console
            print("Average CV Results:")
            print(f"Accuracy : {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall   : {recall:.3f}")
            print(f"ROC-AUC  : {roc_auc:.3f}")

            # Save ONLY the final selected model (Logistic Regression)
            if model_name == "Logistic Regression":

                # Infer model signature for reproducibility
                signature = infer_signature(X, y)

                # Log final model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=cv_results["estimator"][-1],
                    artifact_path="final_model",
                    signature=signature
                )

                # Export model to a stable local directory for API & Docker usage
                mlflow.sklearn.save_model(
                    sk_model=cv_results["estimator"][-1],
                    path="model"
                )


if __name__ == "__main__":
    train_and_evaluate()
