import os

import pandas as pd
import numpy as np
import yaml
import sys

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

def clean_data(df, cols_to_drop):
    """Adjust raw labs/values from MIMIC-IV mini for context of mal-nut"""
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    df['hematocrit_admit'] = df['hematocrit_admit'].fillna(df['hematocrit_calculated_admit'])
    df['marital_status'] = df['marital_status'].fillna('UNKNOWN')

    df = df.drop('hematocrit_calculated_admit', axis=1, errors='ignore')

    df[['systolic','diastolic']] = df["blood_pressure"].str.split('/', expand=True)
    df['systolic'] = pd.to_numeric(df['systolic'], errors='coerce')
    df['diastolic'] = pd.to_numeric(df['diastolic'], errors='coerce')

    df['weight_kg'] = df['weight_lb'] / 2.2
    df['height_cm'] = df['height_in'] * 2.54

    df = df.drop(['blood_pressure','weight_lb','height_in'], axis=1, errors='ignore')

    # df = df.drop('c_reactive_protein_admit',axis=1, errors='ignore')
    
    df['alk_phos_ordered'] = (df["alkaline_phosphatase_admit"].notnull()).astype(int)
    df['ast_ordered'] = (df["asparate_aminotransferase_ast_admit"].notnull()).astype(int)

    # remove heights recorded as < 4 feet (~121 cm)
    invalid_height = df['height_cm'] <= 121
    df.loc[invalid_height, 'height_cm'] = np.nan

    # recalcuate BMI
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100)**2

    return df

class AdmitDataPreprocessor(BaseEstimator, TransformerMixin):
  """Processes raw clinical admit data for use in ML pipeline"""

  def __init__(self, n_neighbors:int = 5):
    self.n_neighbors = n_neighbors
    self.scaler=None
    self.knn_imputer=None
    self.feature_cols=None

  def _reduce_categories(self, X):
    X = X.copy()

    s = X["race"].str.upper()
    X["race_grouped"] = np.select(
        [
            s.str.contains("HISPANIC|LATINO", na=False),
            s.str.startswith("BLACK", na=False),
            s.str.startswith("WHITE", na=False) | s.isin(["PORTUGUESE"]),
            s.isin(["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"])
        ],
        ["hispanic", "black", "white", "unknown/declined"],
        default="other"
    )

    s1 = X['admission_type'].str
    X["admit_type_grouped"] = np.select(
        [
            s1.contains("OBSERVATION"),
            s1.contains("URGENT"),
            s1.contains("EMER"),
            s1.startswith("ELECTIVE"),
            s1.startswith("SURGICAL"),
        ],
        ['observation','urgent_care','emergency','elective','surgical_admit'],
        default='other'
    )

    s2 = X['admission_location'].str
    X['admit_loc_grouped'] = np.select(
        [
            s2.contains("NURSING"),
            s2.contains("ROOM") | s2.contains("PACU") | s2.contains("SITE")| s2.contains("PSYCH"),
            s2.contains("HOSPITAL"),
            s2.startswith("PHYSI") | s2.startswith("CLINIC"),
            s2.startswith("WALK")
        ],
        ['snf','internal','hospital','referral','walkin'],
        default='other_unknown'
    )
    return X

  def fit(self,X,y=None):
    """
    Fit on training data key characteristics to find similar patients to impute missing height
    and weights.
    """
    print('-'*20)
    print('[INFO] Fitting Anthropometic Data')


    X = self._reduce_categories(X)

    matching_features = [
        'age',
        'gender',
        'height_cm',
        'weight_kg',
    ]

    df_encoded = pd.get_dummies(X[matching_features],
                                columns=['gender'],
                                drop_first=True)

    self.feature_cols = df_encoded.columns.tolist()
    print(f'[INFO] Using {self.feature_cols} for KNN:')

    self.scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        self.scaler.fit_transform(df_encoded),
        columns=self.feature_cols,
        index=df_encoded.index
    )

    self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors,
                            weights='distance')

    self.knn_imputer.fit(X_scaled)
    return self

  def transform(self, X):
    """Transforms made from training data"""
    X = self._reduce_categories(X)
    print('-'*20)
    print('[INFO] Starting transform...')
    print('[INFO] Reducing category variety in race, admission_location, and admission_type')

    matching_features = [
        'age',
        'gender',
        'height_cm',
        'weight_kg',
    ]

    df_encoded = pd.get_dummies(X[matching_features],
                                columns=['gender'],
                                drop_first=True)

    for col in self.feature_cols:
      if col not in df_encoded.columns:
        # in test set if cat is missing, will set cat to 0
        df_encoded[col] = 0
    df_encoded = df_encoded[self.feature_cols]

    X_scaled = pd.DataFrame(
    self.scaler.transform(df_encoded),
    columns=self.feature_cols,
    index=df_encoded.index)

    X_imputed = pd.DataFrame(
        self.knn_imputer.transform(X_scaled),
        columns=self.feature_cols,
        index=df_encoded.index
    )

    X_unscaled = pd.DataFrame(
        self.scaler.inverse_transform(X_imputed),
        columns=self.feature_cols,
        index=X.index
    )
    print('[INFO] Imputing missing heights and weights, calculating BMI')

    X['height_cm'] = X_unscaled['height_cm']
    X['weight_kg'] = X_unscaled['weight_kg']

    X['bmi'] = X['weight_kg'] / (X['height_cm'] / 100)**2

    X.drop(['race','admission_type','admission_location'],axis=1, errors='ignore',inplace=True)

    return X

  def fit_transform(self, X, y=None):
    self.fit(X,y)
    return self.transform(X)

def save_processed_df(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def check_data_quality(df, numeric_columns):
    """Return a dictionary of data quality metrics."""
    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mal_pt_data.csv"

    print(f'[INFO] Loading raw MIMIC data from {data_path}...')  # fix: f-string
    df = pd.read_csv(data_path)
    print(f'[INFO] Shape of raw df: {df.shape}')  # fix: f-string

    with open("configs/features.yaml") as f:
        config = yaml.safe_load(f)

    cols_to_drop = config["cols_to_drop"]

    cleaned_df = clean_data(df, cols_to_drop)

    final_features_to_use = config["features_to_use"]

    final_df = cleaned_df[final_features_to_use]

    train, test = train_test_split(final_df,
                                   test_size=0.2,
                                   random_state=42,
                                   stratify=final_df['has_malnutrition'])

    preprocessor = AdmitDataPreprocessor(n_neighbors=5)
    train_prep = preprocessor.fit_transform(train)
    test_prep = preprocessor.transform(test)

    os.makedirs("data/processed", exist_ok=True)
    train_prep.to_csv("data/processed/train.csv", index=False)
    test_prep.to_csv("data/processed/test.csv", index=False)
    print("[INFO] Saved train and test to data/processed/")

    




    

   