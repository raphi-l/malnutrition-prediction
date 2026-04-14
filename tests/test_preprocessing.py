import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")

from preprocessing import clean_data, AdmitDataPreprocessor, check_data_quality


# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def base_df():
    return pd.DataFrame({
        'hematocrit_admit':                        [None,  35.0,  40.0],
        'hematocrit_calculated_admit':             [38.0,  32.0,  None],
        'marital_status':                          [None,  'MARRIED', None],
        'blood_pressure':                          ['120/80', '130/90', None],
        'weight_lb':                               [154.0, 220.0, 110.0],
        'height_in':                               [67.0,  72.0,  47.0],   # last < 4 ft
        'alkaline_phosphatase_admit':              [100.0, None,  200.0],
        'asparate_aminotransferase_ast_admit':     [None,  45.0,  60.0],
        'extra_col':                               [1,     2,     3],
    })


@pytest.fixture
def admit_df():
    """Minimal df shaped for AdmitDataPreprocessor"""
    return pd.DataFrame({
        'age':                [45,   62,   30  ],
        'gender':             ['M',  'F',  'M' ],
        'height_cm':          [175,  None, 160 ],
        'weight_kg':          [80,   70,   None],
        'race':               ['WHITE', 'BLACK OR AFRICAN AMERICAN', 'HISPANIC OR LATINO'],
        'admission_type':     ['EMERGENCY', 'ELECTIVE', 'URGENT'],
        'admission_location': ['WALK IN/SELF REFERRAL', 'HOSPITAL TRANSFER', 'PHYSICIAN REFERRAL'],
    })

class TestCleanData:

    def test_hematocrit_fallback(self, base_df):
        result = clean_data(base_df, [])
        assert result['hematocrit_admit'].iloc[0] == 38.0   # null -> filled
        assert result['hematocrit_admit'].iloc[1] == 35.0   # original kept

    def test_marital_status_unknown_fill(self, base_df):
        result = clean_data(base_df, [])
        assert result['marital_status'].isna().sum() == 0
        assert result['marital_status'].iloc[0] == 'UNKNOWN'
        assert result['marital_status'].iloc[1] == 'MARRIED'

    def test_blood_pressure_split_values(self, base_df):
        result = clean_data(base_df, [])
        assert result['systolic'].iloc[0]  == 120.0
        assert result['diastolic'].iloc[0] == 80.0
        assert result['systolic'].iloc[1]  == 130.0
        assert result['diastolic'].iloc[1] == 90.0

    def test_unit_conversions(self, base_df):
        result = clean_data(base_df, [])
        assert pytest.approx(result['weight_kg'].iloc[0], rel=1e-3) == 154.0 / 2.2
        assert pytest.approx(result['height_cm'].iloc[0], rel=1e-3) == 67.0 * 2.54

    def test_invalid_height_nulled(self, base_df):
        # 47 in * 2.54 = 119.38 cm -> below threshold
        result = clean_data(base_df, [])
        assert pd.isna(result['height_cm'].iloc[2])

    def test_bmi_calculated_correctly(self, base_df):
        result = clean_data(base_df, [])
        w = 154.0 / 2.2
        h = (67.0 * 2.54) / 100
        assert pytest.approx(result['bmi'].iloc[0], rel=1e-3) == w / h**2

# ─────────────────────────────────────────────
# Test Grouping of Categorical Features
# ─────────────────────────────────────────────

class TestReduceCategories:

    @pytest.fixture
    def preprocessor(self):
        return AdmitDataPreprocessor()

    def test_race_grouping(self, preprocessor, admit_df):
        result = preprocessor._reduce_categories(admit_df)
        assert result['race_grouped'].iloc[2] == 'hispanic'
        assert result['race_grouped'].iloc[1] == 'black'
        assert result['race_grouped'].iloc[0] == 'white'

    def test_admit_type_grouping(self, preprocessor, admit_df):
        result = preprocessor._reduce_categories(admit_df)
        assert result['admit_type_grouped'].iloc[0] == 'emergency'
        assert result['admit_type_grouped'].iloc[1] == 'elective'
        assert result['admit_loc_grouped'].iloc[0] == 'walkin'

    def test_admit_loc_grouping(self, preprocessor, admit_df):
        result = preprocessor._reduce_categories(admit_df)
        assert result['admit_loc_grouped'].iloc[1] == 'hospital'
        assert result['admit_loc_grouped'].iloc[2] == 'referral'

# ─────────────────────────────────────────────
# AdmitDataPreprocessor fit/transform
# ─────────────────────────────────────────────

class TestAdmitDataPreprocessor:

    def test_fit_sets_scaler_and_imputer(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        assert p.scaler is not None
        assert p.knn_imputer is not None

    def test_fit_sets_feature_cols(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        assert p.feature_cols is not None
        assert 'height_cm' in p.feature_cols
        assert 'weight_kg' in p.feature_cols

    def test_transform_imputes_missing_height(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        result = p.transform(admit_df)
        assert not result['height_cm'].isna().any()

    def test_transform_imputes_missing_weight(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        result = p.transform(admit_df)
        assert not result['weight_kg'].isna().any()

    def test_transform_recalculates_bmi(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        result = p.transform(admit_df)
        assert 'bmi' in result.columns
        assert not result['bmi'].isna().any()

    def test_transform_drops_raw_categorical_cols(self, admit_df):
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        result = p.transform(admit_df)
        for col in ['race', 'admission_type', 'admission_location']:
            assert col not in result.columns

    def test_transform_handles_missing_gender_category(self, admit_df):
        """If test set lacks a gender seen in train, column should default to 0"""
        p = AdmitDataPreprocessor()
        p.fit(admit_df)
        test_df = admit_df.copy()
        test_df['gender'] = 'M'   # only one gender in test
        result = p.transform(test_df)
        assert result is not None

    def test_fit_transform_matches_fit_then_transform(self, admit_df):
        p1 = AdmitDataPreprocessor()
        result1 = p1.fit_transform(admit_df.copy())

        p2 = AdmitDataPreprocessor()
        p2.fit(admit_df.copy())
        result2 = p2.transform(admit_df.copy())

        pd.testing.assert_frame_equal(result1.reset_index(drop=True),
                                      result2.reset_index(drop=True))


# ─────────────────────────────────────────────
# check_data_quality
# ─────────────────────────────────────────────

class TestCheckDataQuality:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'a': [1.0, 2.0, None],
            'b': [4.0, 5.0, 6.0],
        })

    def test_total_rows(self, sample_df):
        report = check_data_quality(sample_df, [])
        assert report['total_rows'] == 3

    def test_total_nulls(self, sample_df):
        report = check_data_quality(sample_df, [])
        assert report['total_nulls'] == 1

    def test_null_percentage(self, sample_df):
        report = check_data_quality(sample_df, [])
        assert pytest.approx(report['null_percentage'], abs=0.01) == 16.67

    def test_duplicate_rows(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
        report = check_data_quality(df, [])
        assert report['duplicate_rows'] == 1

    def test_numeric_col_range(self, sample_df):
        report = check_data_quality(sample_df, ['a', 'b'])
        assert report['a_min'] == 1.0
        assert report['a_max'] == 2.0   # NaN excluded by min/max
        assert report['b_min'] == 4.0
        assert report['b_max'] == 6.0

    def test_missing_numeric_col_skipped(self, sample_df):
        report = check_data_quality(sample_df, ['nonexistent'])
        assert 'nonexistent_min' not in report