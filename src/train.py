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

    lightgbm_params = model_config["lgb.LGBMClassifier"]
    quality_config = model_config["model_quality"]

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

    report = classification_report(y_test, y_pred, output_dict=True)
   
    metrics = {
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report["1"]["precision"], 4),  # for the positive class
        "recall":    round(report["1"]["recall"], 4),
        "f1_score":  round(report["1"]["f1-score"], 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_proba), 4),
    }

    # Check thresholds
    if metrics["accuracy"] < quality_config["min_accuracy"]:
        print(f"\nWARNING: Accuracy {metrics['accuracy']} is below threshold {quality_config['min_accuracy']}")
    if metrics["f1_score"] <= quality_config["min_f1"]:
        print(f"\nWARNING: F1 {metrics['f1_score']} is at/below threshold {quality_config['min_f1']}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics

if __name__ == "__main__":
    
    try:
        data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError
    except (IndexError, FileNotFoundError):
        data_path = 'https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_train_sample.csv'
    
    df = load_data(data_path)  
    metrics = train_model(df)

    with open("configs/model_params.yaml") as f:
        model_config = yaml.safe_load(f)

    quality_config = model_config["model_quality"]

    # Exit with error if thresholds not met
    if metrics["accuracy"] < quality_config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < quality_config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")

