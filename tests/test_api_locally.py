import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "uv_app"))

from api import app

client = TestClient(app)

HTTP_STATUS_CODE_OK = 200
HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY = 422


@pytest.fixture
def test_data() -> dict:
    return {
        "age": 35,
        "employment_type": "Private Sector/Self Employed",
        "graduate_or_not": "Yes",
        "annual_income": 600000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
        "ever_travelled_abroad": "Yes",
    }


def test_predict_valid_input(test_data: dict) -> None:
    """Test the predict endpoint with valid input data."""
    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_OK

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["probability"] <= 1


def test_predict_invalid_age(test_data: dict) -> None:
    """Test the predict endpoint with invalid age."""
    test_data["age"] = 5
    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_invalid_employment_type(test_data: dict) -> None:
    """Test the predict endpoint with invalid employment type."""
    test_data["employment_type"] = "Invalid Type"
    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_invalid_annual_income(test_data: dict) -> None:
    """Test the predict endpoint with invalid annual income."""
    test_data["annual_income"] = -1000
    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_missing_field(test_data: dict) -> None:
    """Test the predict endpoint with a missing required field."""
    del test_data["ever_travelled_abroad"]
    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY
