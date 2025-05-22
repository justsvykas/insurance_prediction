import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from schemas import CustomerRequest, CustomerResponse
from settings import server_settings

app = FastAPI(
    title="Travel Insurance Prediction API",
    description="API for predicting customer likelihood to purchase travel insurance",
    version="1.0.0",
)

with (Path(__file__).parent / "full_pipeline.pkl").open("rb") as f:
    pipeline = pickle.load(f)  # noqa: S301


@app.post("/predict")
async def predict(customer: CustomerRequest) -> CustomerResponse:
    """Predict the likelihood of a customer to purchase travel insurance."""
    input_df = pd.DataFrame(
        [
            {
                "Age": customer.age,
                "Employment Type": customer.employment_type,
                "GraduateOrNot": customer.graduate_or_not,
                "AnnualIncome": customer.annual_income,
                "FamilyMembers": customer.family_members,
                "ChronicDiseases": customer.chronic_diseases,
                "FrequentFlyer": customer.frequent_flyer,
                "EverTravelledAbroad": customer.ever_travelled_abroad,
            }
        ]
    )
    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)[0][1]
    prediction_str = "Yes" if prediction[0] == 1 else "No"
    return CustomerResponse(prediction=prediction_str, probability=float(probability))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=server_settings.HOST,
        port=server_settings.PORT,
        log_level=server_settings.LOG_LEVEL,
    )
