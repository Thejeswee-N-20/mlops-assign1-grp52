# loads the trained MLflow model and performs inference. It will be reused later for API serving.

import mlflow.pyfunc
import pandas as pd


def load_model(model_uri: str):
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction
