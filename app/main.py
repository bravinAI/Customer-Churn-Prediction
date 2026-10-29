"""FastAPI wrapper around the saved customer churn pipeline."""

from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

EXPECTED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Saved churn model was not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
if not isinstance(model, Pipeline):
    raise TypeError(f"Expected a scikit-learn Pipeline, got {type(model).__name__}")
check_is_fitted(model)

MODEL_FEATURES = list(model.feature_names_in_)
if MODEL_FEATURES != EXPECTED_FEATURES:
    raise RuntimeError(
        "Saved model feature schema does not match the API schema. "
        f"Model: {MODEL_FEATURES}; API: {EXPECTED_FEATURES}"
    )


NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


class CustomerFeatures(BaseModel):
    """Required model inputs. `TotalCharges` may be null and is imputed by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    gender: NonEmptyString
    SeniorCitizen: int = Field(strict=True, ge=0, le=1)
    Partner: NonEmptyString
    Dependents: NonEmptyString
    tenure: int = Field(strict=True, ge=0, le=72)
    PhoneService: NonEmptyString
    MultipleLines: NonEmptyString
    InternetService: NonEmptyString
    OnlineSecurity: NonEmptyString
    OnlineBackup: NonEmptyString
    DeviceProtection: NonEmptyString
    TechSupport: NonEmptyString
    StreamingTV: NonEmptyString
    StreamingMovies: NonEmptyString
    Contract: NonEmptyString
    PaperlessBilling: NonEmptyString
    PaymentMethod: NonEmptyString
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class PredictionResponse(BaseModel):
    predicted_churn: Literal["No", "Yes"]
    churn_probability: float = Field(ge=0, le=1)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn using the fitted pipeline from the existing analysis.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API is running and the fitted pipeline loaded."""
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Return the predicted class and the pipeline's churn probability."""
    row = features.model_dump()
    if row["TotalCharges"] is None:
        row["TotalCharges"] = np.nan

    input_frame = pd.DataFrame([row], columns=MODEL_FEATURES)
    predicted_class = int(model.predict(input_frame)[0])
    churn_probability = float(model.predict_proba(input_frame)[0, 1])

    return PredictionResponse(
        predicted_churn="Yes" if predicted_class == 1 else "No",
        churn_probability=churn_probability,
    )
