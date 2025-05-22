import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "uv_app"))

from api import app

client = TestClient(app)

HTTP_STATUS_CODE_OK = 200
HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY = 422


def test_predict_valid_input() -> None:
    """Test the predict endpoint with valid input data."""
    test_data = {
        "age": 35,
        "employment_type": "Private Sector/Self Employed",
        "graduate_or_not": "Yes",
        "annual_income": 600000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
        "ever_travelled_abroad": "Yes",
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_OK

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["probability"] <= 1


def test_predict_invalid_age() -> None:
    """Test the predict endpoint with invalid age."""
    test_data = {
        "age": 5,
        "employment_type": "Private Sector/Self Employed",
        "graduate_or_not": "Yes",
        "annual_income": 600000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
        "ever_travelled_abroad": "Yes",
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_invalid_employment_type() -> None:
    """Test the predict endpoint with invalid employment type."""
    test_data = {
        "age": 35,
        "employment_type": "Invalid Type",
        "graduate_or_not": "Yes",
        "annual_income": 600000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
        "ever_travelled_abroad": "Yes",
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_invalid_annual_income() -> None:
    """Test the predict endpoint with invalid annual income."""
    test_data = {
        "age": 35,
        "employment_type": "Private Sector/Self Employed",
        "graduate_or_not": "Yes",
        "annual_income": -1000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
        "ever_travelled_abroad": "Yes",
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY


def test_predict_missing_field() -> None:
    """Test the predict endpoint with a missing required field."""
    test_data = {
        "age": 35,
        "employment_type": "Private Sector/Self Employed",
        "graduate_or_not": "Yes",
        "annual_income": 600000,
        "family_members": 4,
        "chronic_diseases": 0,
        "frequent_flyer": "Yes",
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == HTTP_STATUS_CODE_UNPROCESSABLE_ENTITY
