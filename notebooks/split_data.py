from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

raw_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "tejashvi14/travel-insurance-prediction-data",
    "TravelInsurancePrediction.csv",
)

Path("data").mkdir(parents=True, exist_ok=True)

train_df, temp_df = train_test_split(
    raw_df, test_size=0.3, random_state=42, stratify=raw_df["TravelInsurance"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["TravelInsurance"],
)

raw_df.to_csv("data/raw_travel_insurance.csv", index=False)
train_df.to_csv("data/train_travel_insurance.csv", index=False)
val_df.to_csv("data/val_travel_insurance.csv", index=False)
test_df.to_csv("data/test_travel_insurance.csv", index=False)

print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")
