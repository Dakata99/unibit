#!/usr/bin/env python3

"""
https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset

Problem area: liver disease states

Ranking:
    Top-tier (keep 100%): ALT, AST, Bilirubin, Albumin, Alk_Phosphatase, Platelets,
                          All symptom columns, Comorbidities, Alcohol_Consumption, Age, Gender

    Mid-tier (for richer models): BMI, Obesity_Class, Diet_Quality,
                                  Physical_Activity, Sleep_Hours, Smoking_Status

    Low-tier (drop or test with feature importance): Occupation, Patient_ID

Training 1:
Make symptoms less important (patients lie 😅). Keep lab results as must have.
Make 2 subnetworks:
    [Symptoms] → small dense block
    [Labs + demographics] → larger dense block → concatenate → final layers → output

Training 2: make everythin important.
Training 3: rely only on lab tests, drop everything else.
"""

import kagglehub
import pandas as pd
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("paramjeetsinghds/indian-liver-disease-dataset")
print("Dataset downloaded to:", path)

# Load the CSV (adjust the filename if needed)
df = pd.read_csv(Path(path) / "Training_indian_liver_disease_dataset.csv")

print(df.head())
print(df.info())
# for feature in df.columns:
#     print(f"Feature {feature} classes: {df[feature].unique()}")

print(f"Features: {df.columns}")
print(f"Target: {df["Liver_Disease_Type"]}")
print(f"Target: {df["Liver_Disease_Type"].unique()}")
print(df[])

# print('--------------------------------------------------------------')
# df = df[~df["Liver_Disease_Type"].isin(["Healthy", "Fatty_Liver"])]
# print(df)
