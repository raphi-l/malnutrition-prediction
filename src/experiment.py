import pandas as pd
import numpy as np
import json
import os
import sys
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import lightgbm as lgb
import mlflow

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

    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    mlflow.set_experiment("malnutrition-prediction-lightgbm")

    with mlflow.start_run():

        # ── Log params ──
        mlflow.log_param("n_estimators",  lightgbm_params['n_estimators'])
        mlflow.log_param("max_depth",     lightgbm_params['max_depth'])
        mlflow.log_param("learning_rate", lightgbm_params['learning_rate'])
        mlflow.log_param("class_weight",  lightgbm_params['class_weight'])
        mlflow.log_param("test_size",     lightgbm_params['test_size'])
        mlflow.log_param("random_state",  lightgbm_params['random_state'])
        mlflow.log_param("n_features",    X_train.shape[1])

        # ── Train ──
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

        # ── Evaluate ──
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("\nValidation Set Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "accuracy":  round(report["accuracy"], 4),
            "precision": round(report["1"]["precision"], 4),
            "recall":    round(report["1"]["recall"], 4),
            "f1_score":  round(report["1"]["f1-score"], 4),
            "roc_auc":   round(roc_auc_score(y_test, y_pred_proba), 4),
        }

        # ── Log metrics ──
        mlflow.log_metrics(metrics)

        # ── Quality warnings ──
        if metrics["accuracy"] < quality_config["min_accuracy"]:
            print(f"\nWARNING: Accuracy {metrics['accuracy']} below threshold {quality_config['min_accuracy']}")
        if metrics["f1_score"] <= quality_config["min_f1"]:
            print(f"\nWARNING: F1 {metrics['f1_score']} at/below threshold {quality_config['min_f1']}")

        # ── Log model ──
        mlflow.lightgbm.log_model(model, "model")

        # ── Log config as artifact ──
        with open("config_snapshot.json", "w") as f:
            json.dump(lightgbm_params, f, indent=2)
        mlflow.log_artifact("config_snapshot.json")
        os.remove("config_snapshot.json")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View in UI: mlflow ui")

        # ── Save model and metrics locally ──
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/results.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

def load_data(path):
    """Load CSV data and return features and labels"""
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test_mini.csv"

    df = load_data(data_path) 
    metrics = train_model(df)