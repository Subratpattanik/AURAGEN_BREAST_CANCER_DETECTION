
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
metadata_path = BASE_DIR / "metadata.csv"

# Load metadata
df = pd.read_csv(metadata_path)

# Check patient labels
patient_data = df.groupby("patient_id")["label"].nunique()

if (patient_data != 1).any():
    raise ValueError("Some patients have multiple labels!")

# One row per patient
patients = (
    df[["patient_id", "label"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# First split: 85% remaining, 15% test
train_val, test = train_test_split(
    patients,
    test_size=0.15,
    random_state=42,
    stratify=patients["label"]
)

# Second split: approximately 70% train, 15% validation
train, val = train_test_split(
    train_val,
    test_size=0.1765,
    random_state=42,
    stratify=train_val["label"]
)

# Assign every image using its patient ID
train_df = df[df["patient_id"].isin(train["patient_id"])]
val_df = df[df["patient_id"].isin(val["patient_id"])]
test_df = df[df["patient_id"].isin(test["patient_id"])]

# Save CSV files
train_df.to_csv(BASE_DIR / "train.csv", index=False)
val_df.to_csv(BASE_DIR / "validation.csv", index=False)
test_df.to_csv(BASE_DIR / "test.csv", index=False)

# Report
print("TRAINING:", len(train_df), "images")
print("VALIDATION:", len(val_df), "images")
print("TESTING:", len(test_df), "images")

print("\nPatient counts:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nClass distribution:")
print("Train:\n", train["label"].value_counts())
print("Validation:\n", val["label"].value_counts())
print("Test:\n", test["label"].value_counts())

# Verify patient separation
assert set(train.patient_id).isdisjoint(val.patient_id)
assert set(train.patient_id).isdisjoint(test.patient_id)
assert set(val.patient_id).isdisjoint(test.patient_id)

print("\nPATIENT LEAKAGE CHECK: PASSED")
