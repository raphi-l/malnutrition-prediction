import pandas as pd
import numpy as np
import os
import sys
import yaml
import json

sys.path.insert(0, "src")
from train import load_data, train_model

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score

def load_model():
    model_path = "models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_data(path):
    """Load CSV data and return features and labels"""
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def eval(df, model):
    
    X = df.drop('has_malnutrition', axis=1)
    y = df['has_malnutrition']

    # Convert categorical to category dtype for LightGBM
    # Keeping this out of load_data to allow use with other model types
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y, y_pred_proba):.3f}")

    report = classification_report(y, y_pred, output_dict=True)
   
    metrics = {
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report["1"]["precision"], 4),  # for the positive class
        "recall":    round(report["1"]["recall"], 4),
        "f1_score":  round(report["1"]["f1-score"], 4),
        "roc_auc":   round(roc_auc_score(y, y_pred_proba), 4),
    }

    # Check thresholds
    if metrics["accuracy"] < quality_config["min_accuracy"]:
        print(f"\nWARNING: Accuracy {metrics['accuracy']} is below threshold {quality_config['min_accuracy']}")
    if metrics["f1_score"] <= quality_config["min_f1"]:
        print(f"\nWARNING: F1 {metrics['f1_score']} is at/below threshold {quality_config['min_f1']}")

    return metrics

if __name__ == "__main__":  
    
    try:
        data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError
    except (IndexError, FileNotFoundError):
        data_path = 'https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_test.csv'
    
    df = load_data(data_path)  
    
    with open("configs/model_params.yaml") as f:
        model_config = yaml.safe_load(f)
    quality_config = model_config["model_quality"]

    model = load_model()

    test_metrics = eval(df, model)

    #pprint(test_metrics)

    with open("metrics/results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Exit with error if thresholds not met
    if test_metrics["accuracy"] < quality_config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold on test data")
        sys.exit(1)
    if test_metrics["f1_score"] < quality_config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold on test data")
        sys.exit(1)

    print("\nAll thresholds passed!")