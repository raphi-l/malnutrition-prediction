import pytest
import pandas as pd
import numpy as np
import os
import sys

@pytest.fixture
def train_data_df():
    local_path = "data/processed/train.csv"
    fallback_url = 'https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_train_sample.csv'

    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    
    return pd.read_csv(fallback_url)

@pytest.fixture
def expected_numeric():
    return {
        "age": range(18, 99),
        "height_cm": (120, 210),
        "weight_kg": (30, 300),
        "bmi": (10, 80),
        "glucose_admit": (20, 2000),
        "hematocrit_admit": (3, 70),
        "hemoglobin_admit": (1.5, 23),
        "potassium_admit": (1.0, 10.0),
        "systolic": (40, 280),
        "diastolic": (15, 180),
        "alk_phos_ordered": (0, 1), #boolean
        "ast_ordered": (0, 1), # boolean
    }

@pytest.fixture
def expected_demographics():
    return {
        "gender": ["M", "F"],
        "marital_status": ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED", "UNKNOWN"],
        "race_grouped": ["hispanic", "black", "white", "unknown/declined", "other"],
        "admit_type_grouped": ["observation", "urgent_care", "emergency", "elective", "surgical_admit", "other"],
        "admit_loc_grouped": ["snf", "internal", "hospital", "referral", "walkin", "other_unknown"],
    }


# ─────────────────────────────────────────────
# Test target(s)
# ─────────────────────────────────────────────

class TestTargets:

    def test_malnutrition(self, train_data_df):
        unique_encoded = set(train_data_df["has_malnutrition"].unique())
        assert unique_encoded.issubset({0, 1}), \
            f"[INFO] Invalid values in 'has_malnutrition' column. Found {unique_encoded}!"


# ─────────────────────────────────────────────
# Test numeric values
# ─────────────────────────────────────────────

class TestNumeric:

    ANTHROPOMETRIC_COLS = {"age", "height_cm", "weight_kg", "bmi"}

    def test_anthropometric(self, train_data_df, expected_numeric):
        for col in self.ANTHROPOMETRIC_COLS:
            bounds = expected_numeric[col]
            low = bounds.start if isinstance(bounds, range) else bounds[0]
            high = bounds.stop if isinstance(bounds, range) else bounds[1]
            assert train_data_df[col].between(low, high).all(), \
                f"{col} has values outside expected range"

    def test_labs(self, train_data_df, expected_numeric):
        lab_cols = {k for k in expected_numeric if k not in self.ANTHROPOMETRIC_COLS}
        for col in lab_cols:
            low, high = expected_numeric[col]
            col_data = train_data_df[col].dropna()
            assert col_data.between(low, high).all(), \
                f"{col} has values outside expected range"


# ─────────────────────────────────────────────
# Test categorical values
# ─────────────────────────────────────────────

class TestCategorical:

    def test_demographics(self, train_data_df, expected_demographics):
        for col in expected_demographics:
            assert col in train_data_df.columns.tolist(), \
                f"[INFO] Missing {col} column!"

            unique_entries = set(train_data_df[col].unique())
            expected_entries = set(expected_demographics[col])
            unexpected_values = unique_entries - expected_entries

            assert not unexpected_values, \
                f"[INFO] Unexpected values in {col} column. Found: {unexpected_values}"