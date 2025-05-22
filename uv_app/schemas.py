from typing import Literal

from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer's age in years",
        json_schema_extra={"example": 35},
    )

    employment_type: Literal["Private Sector/Self Employed", "Government Sector"] = (
        Field(
            ...,
            description="Type of employment",
            json_schema_extra={"example": "Private Sector/Self Employed"},
        )
    )

    graduate_or_not: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has graduated",
        json_schema_extra={"example": "Yes"},
    )

    annual_income: float = Field(
        ...,
        ge=0,
        description="Annual income in USD",
        json_schema_extra={"example": 600000},
    )

    family_members: int = Field(
        ...,
        ge=1,
        le=50,
        description="Number of family members",
        json_schema_extra={"example": 4},
    )

    chronic_diseases: Literal[0, 1] = Field(
        ...,
        description="Whether the customer has chronic diseases (0=No, 1=Yes)",
        json_schema_extra={"example": 0},
    )

    frequent_flyer: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer is a frequent flyer",
        json_schema_extra={"example": "Yes"},
    )

    ever_travelled_abroad: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has ever travelled abroad",
        json_schema_extra={"example": "Yes"},
    )


class CustomerResponse(BaseModel):
    prediction: Literal["Yes", "No"] = Field(
        ...,
        description="Prediction for travel insurance",
        json_schema_extra={"example": "Yes"},
    )

    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of the prediction",
        json_schema_extra={"example": 0.95},
    )
