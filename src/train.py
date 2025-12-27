# trains multiple classification models on the heart disease dataset and evaluates them using cross-validation.

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from data_loader import load_data
from features import create_preprocessing_pipeline



def train_and_evaluate():
    # Load the processed dataset
    df = load_data("data/processed/heart_cleaned.csv")

    # Separate features and target variable
    X = df.drop("target", axis=1)
    y = df["target"]

    # All features are numerical in this dataset
    numerical_features = X.columns.tolist()

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_features)

    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # Evaluation metrics
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    # Set MLflow experiment name
    mlflow.set_experiment("Heart Disease Classification")

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining and evaluating: {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Log model parameters
            if model_name == "Logistic Regression":
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", model.max_iter)
                
            if model_name == "Random Forest":
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("random_state", model.random_state)
                
            # Create full pipeline
            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
            
            # Perform cross-validation
            cv_results = cross_validate(
                clf,
                X,
                y,
                cv=5,
                scoring=scoring,
                return_estimator=True
            )
            
            # Calculate mean metrics
            acc = cv_results["test_accuracy"].mean()
            prec = cv_results["test_precision"].mean()
            rec = cv_results["test_recall"].mean()
            roc = cv_results["test_roc_auc"].mean()
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", roc)
            
            # Infer model signature for reproducibility
            signature = infer_signature(X, y)
            
            # Log the final model (including preprocessing pipeline)
            mlflow.sklearn.log_model(
                sk_model=cv_results["estimator"][-1],
                artifact_path="final_model",
                signature=signature
                )
            
            print("Average CV Results:")
            print(f"Accuracy : {acc:.3f}")
            print(f"Precision: {prec:.3f}")
            print(f"Recall   : {rec:.3f}")
            print(f"ROC-AUC  : {roc:.3f}")


if __name__ == "__main__":
    train_and_evaluate()