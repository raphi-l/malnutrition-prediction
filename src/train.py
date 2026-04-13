import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import csv
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score

import lightgbm as lgb

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))

def load_data(path):
    """Load CSV data and return features and labels"""
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def train_model(df, config=None):
    
    with open("configs/model_params.yaml") as f:
        model_config = yaml.safe_load(f)

    lightgbm_params = model_config["lightgbm"]

    X = df.drop('has_malnutrition', axis=1)
    y = df['has_malnutrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=lightgbm_params['test_size'],
        random_state=lightgbm_params['random_state'],
        stratify=y
    )

    # Convert categorical to category dtype for LightGBM
    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=lightgbm_params['n_estimators'],
        max_depth=lightgbm_params['max_depth'],
        learning_rate=lightgbm_params['learning_rate'],
        random_state=lightgbm_params['random_state'],
        class_weight=lightgbm_params['class_weight'],
        verbose=-1
    )
    categorical_features = X_train.select_dtypes(include='category').columns.tolist()
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    # Evaluate

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nValidation Set Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test_mini.csv"

    df = load_data(data_path)  # Adjust path as needed
    train_model(df)



