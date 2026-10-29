"""Verification tests for the existing fitted churn model API."""

import asyncio
import unittest
from pathlib import Path

import httpx
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from app.main import EXPECTED_FEATURES, MODEL_PATH, app, model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def api_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def make_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(make_request())


def first_dataset_customer() -> dict:
    """Build an API request from the first existing CSV row, excluding ID and target."""
    data = pd.read_csv(DATASET_PATH)
    row = data.iloc[0]
    request = {feature: row[feature] for feature in EXPECTED_FEATURES}
    request["SeniorCitizen"] = int(request["SeniorCitizen"])
    request["tenure"] = int(request["tenure"])
    request["MonthlyCharges"] = float(request["MonthlyCharges"])
    total_charges = pd.to_numeric(request["TotalCharges"], errors="coerce")
    request["TotalCharges"] = None if pd.isna(total_charges) else float(total_charges)
    return request


class ChurnApiTests(unittest.TestCase):
    def test_saved_model_loads_as_fitted_pipeline_with_expected_features(self) -> None:
        self.assertTrue(MODEL_PATH.is_file())
        self.assertIsInstance(model, Pipeline)
        check_is_fitted(model)
        self.assertEqual(list(model.feature_names_in_), EXPECTED_FEATURES)

    def test_health_endpoint(self) -> None:
        response = api_request("GET", "/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": True})

    def test_valid_dataset_row_returns_prediction_and_probability(self) -> None:
        request = first_dataset_customer()
        response = api_request("POST", "/predict", json=request)
        self.assertEqual(response.status_code, 200, response.text)

        result = response.json()
        self.assertIn(result["predicted_churn"], {"No", "Yes"})
        self.assertGreaterEqual(result["churn_probability"], 0.0)
        self.assertLessEqual(result["churn_probability"], 1.0)

        model_input = pd.DataFrame([request], columns=EXPECTED_FEATURES)
        expected_class = int(model.predict(model_input)[0])
        expected_probability = float(model.predict_proba(model_input)[0, 1])
        self.assertEqual(result["predicted_churn"], "Yes" if expected_class else "No")
        self.assertAlmostEqual(result["churn_probability"], expected_probability, places=7)

    def test_pipeline_imputes_missing_total_charges(self) -> None:
        request = first_dataset_customer()
        request["TotalCharges"] = None
        response = api_request("POST", "/predict", json=request)
        self.assertEqual(response.status_code, 200, response.text)

    def test_missing_required_feature_is_rejected(self) -> None:
        request = first_dataset_customer()
        request.pop("tenure")
        response = api_request("POST", "/predict", json=request)
        self.assertEqual(response.status_code, 422)

    def test_invalid_feature_type_is_rejected(self) -> None:
        request = first_dataset_customer()
        request["SeniorCitizen"] = "sometimes"
        response = api_request("POST", "/predict", json=request)
        self.assertEqual(response.status_code, 422)

    def test_customer_id_and_target_are_not_accepted_as_inputs(self) -> None:
        request = first_dataset_customer()
        request["customerID"] = "not-an-input"
        request["Churn"] = "No"
        response = api_request("POST", "/predict", json=request)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
