"""
FastAPI application to serve the trained Heart Disease prediction model.
Includes prediction, confidence estimation, request logging,
and a simple monitoring endpoint.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import math
import logging
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load trained sklearn model (pipeline saved via MLflow)
model = mlflow.sklearn.load_model("model")

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simple in-memory monitoring metrics
REQUEST_COUNT = 0
SERVICE_START_TIME = datetime.utcnow()


# Define input schema using Pydantic
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "API is running"}


@app.post("/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    """
    Predict heart disease risk based on patient data.
    Returns binary prediction and confidence score.
    """

    global REQUEST_COUNT
    REQUEST_COUNT += 1

    # Convert incoming JSON to pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Get decision score from Logistic Regression
    score = model.decision_function(input_df)[0]

    # Convert score to probability-like confidence using sigmoid
    confidence = 1 / (1 + math.exp(-score))

    # Final prediction
    prediction = int(confidence >= 0.5)

    # Log the request
    logging.info(
        f"Prediction request | Prediction={prediction} | Confidence={confidence:.4f}"
    )

    return {
        "prediction": prediction,
        "confidence": float(confidence)
    }


@app.get("/metrics")
def metrics():
    """
    Simple monitoring endpoint to expose API usage metrics.
    """
    return {
        "total_prediction_requests": REQUEST_COUNT,
        "service_start_time": SERVICE_START_TIME.isoformat()
    }
