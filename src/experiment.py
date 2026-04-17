import pandas as pd
import numpy as np
import json
import os
import sys
import yaml
import pickle
from sklearn.model_selection import ParameterSampler, train_test_split
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

    with mlflow.start_run(nested=False) as parent_run:

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

        categorical_features = X_train.select_dtypes(include='category').columns.tolist()
        
        fit_params = model_config.get('fit_params', {}) or {}
        fit_params = {**fit_params, "categorical_feature": categorical_features}

        best_model = None
        best_metric = -np.inf
        best_search_params = {}

        if search_config.get('type') == 'randomized' and param_distributions:
            candidate_params = list(ParameterSampler(
                param_distributions,
                n_iter=search_n_iter,
                random_state=search_random_state
            ))

            for idx, params in enumerate(candidate_params, start=1):
                with mlflow.start_run(nested=True) as candidate_run:
                    mlflow.set_tag("search_iteration", idx)
                    mlflow.log_params(params)
                    mlflow.log_param("search_type", "randomized")
                    mlflow.log_param("candidate_index", idx)

                    candidate_args = {
                        **{
                            "n_estimators": lightgbm_params.get('n_estimators', 100),
                            "max_depth": lightgbm_params.get('max_depth', -1),
                            "learning_rate": lightgbm_params.get('learning_rate', 0.1),
                            "class_weight": lightgbm_params.get('class_weight', None),
                        },
                        **params,
                    }
                    candidate_args["random_state"] = random_state
                    candidate_args["verbose"] = -1

                    candidate_model = lgb.LGBMClassifier(**candidate_args)
                    candidate_model.fit(X_train, y_train, **fit_params)

                    y_pred = candidate_model.predict(X_test)
                    y_pred_proba = candidate_model.predict_proba(X_test)[:, 1]
                    report = classification_report(y_test, y_pred, output_dict=True)
                    candidate_metrics = {
                        "accuracy":  round(report["accuracy"], 4),
                        "precision": round(report["1"]["precision"], 4),
                        "recall":    round(report["1"]["recall"], 4),
                        "f1_score":  round(report["1"]["f1-score"], 4),
                        "roc_auc":   round(roc_auc_score(y_test, y_pred_proba), 4),
                    }
                    mlflow.log_metrics(candidate_metrics)

                    metric_value = candidate_metrics.get(search_scoring, candidate_metrics.get("recall", 0))
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_model = candidate_model
                        best_search_params = params

            if best_model is None:
                raise RuntimeError("Random search did not evaluate any candidates.")

            model = best_model
            mlflow.log_param("best_search_scoring", search_scoring)
            mlflow.log_param("best_search_value", best_metric)
            mlflow.log_params({f"best_{k}": v for k, v in best_search_params.items()})
        else:
                    # ── Train ──
            model = lgb.LGBMClassifier(
                n_estimators=lightgbm_params.get('n_estimators', 100),
                max_depth=lightgbm_params.get('max_depth', -1),
                learning_rate=lightgbm_params.get('learning_rate', 0.1),
                random_state=random_state,
                class_weight=lightgbm_params.get('class_weight', None),
                verbose=-1
            )
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