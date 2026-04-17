import pandas as pd
import numpy as np
import json
import os
import sys
import yaml
import pickle
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import lightgbm as lgb
import mlflow

def train_model(df, config=None):
    
    with open("configs/model_params.yaml") as f:
        model_config = yaml.safe_load(f)

    lightgbm_params = model_config["lgb.LGBMClassifier"]
    
    quality_config = model_config["model_quality"]
    
    search_config = model_config.get("search", {})
    
    split_config = model_config.get("split", {})
    param_distributions = model_config.get("param_distributions", {})
    
    target_column = model_config.get("target", lightgbm_params.get("target", "has_malnutrition"))
    
    random_state = lightgbm_params.get("random_state", lightgbm_params.get("random_seed", 42))

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=lightgbm_params['test_size'],
        random_state=random_state,
        stratify=y
    )

    for col in X_train.select_dtypes(include='object').columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    mlflow.set_experiment("malnutrition-prediction-lightgbm")

    with mlflow.start_run():

        # ── Log params ──
        mlflow.log_param("test_size",     lightgbm_params['test_size'])
        mlflow.log_param("random_state",  random_state)
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_param("search_type",   search_config.get('type', 'none'))

        search_n_iter = search_config.get('n_iter', 30)
        search_scoring = search_config.get('scoring', 'recall')
        search_random_state = search_config.get('random_state', random_state)
        search_n_jobs = search_config.get('n_jobs', -1)
        search_verbose = search_config.get('verbose', 1)

        if search_config.get('type') == 'randomized':
            mlflow.log_param("n_iter", search_n_iter)
            mlflow.log_param("scoring", search_scoring)

        # ── Train ──
        model = lgb.LGBMClassifier(
            n_estimators=lightgbm_params.get('n_estimators', 100),
            max_depth=lightgbm_params.get('max_depth', -1),
            learning_rate=lightgbm_params.get('learning_rate', 0.1),
            random_state=random_state,
            class_weight=lightgbm_params.get('class_weight', None),
            verbose=-1
        )
        categorical_features = X_train.select_dtypes(include='category').columns.tolist()
        fit_params = model_config.get('fit_params', {}) or {}
        fit_params = {**fit_params, "categorical_feature": categorical_features}

        if search_config.get('type') == 'randomized' and param_distributions:
            cv = StratifiedKFold(
                n_splits=split_config.get('n_splits', 5),
                shuffle=split_config.get('shuffle', True),
                random_state=split_config.get('random_state', random_state)
            ) if split_config.get('type') == 'stratified_kfold' else split_config.get('n_splits', 5)

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=search_n_iter,
                scoring=search_scoring,
                cv=cv,
                random_state=search_random_state,
                n_jobs=search_n_jobs,
                verbose=search_verbose,
                refit=True
            )
            search.fit(X_train, y_train, **fit_params)
            model = search.best_estimator_
            best_params = search.best_params_
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        else:
            model.fit(X_train, y_train, **fit_params)

        # ── Evaluate ──
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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