import sys
import mlflow
import yaml
import pandas as pd

DEFAULT_EXPERIMENT = "malnutrition-prediction-lightgbm"

def get_experiment_by_name(name: str):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise SystemExit(
            f"Experiment '{name}' not found. "
            f"Try using an existing MLflow experiment name or start with '{DEFAULT_EXPERIMENT}'."
        )
    return experiment


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERIMENT
    experiment = get_experiment_by_name(experiment_name)

    # Get all finished runs, including nested child runs from search
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        run_view_type=mlflow.entities.ViewType.ALL
    )

    if runs.empty:
        raise SystemExit(
            f"No finished runs found for experiment '{experiment_name}'. "
            "Check that the experiment has completed MLflow runs."
        )

    # Sort by F1 score descending (balancing precision and recall)
    runs = runs.sort_values('metrics.f1_score', ascending=False)

    print("Top 5 Runs by F1 Score:")
    print("=" * 80)
    for i, row in runs.head(5).iterrows():
        print(f"\nRun: {row['run_id'][:8]}...")
        recall = row.get('metrics.recall', 'N/A')
        f1 = row.get('metrics.f1_score', 'N/A')
        accuracy = row.get('metrics.accuracy', 'N/A')
        auc_roc = row.get('metrics.roc_auc', 'N/A')
        
        print(f"  Recall:    {recall if isinstance(recall, str) else f'{recall:.4f}'}")
        print(f"  F1:        {f1 if isinstance(f1, str) else f'{f1:.4f}'}")
        print(f"  Accuracy:  {accuracy if isinstance(accuracy, str) else f'{accuracy:.4f}'}")
        print(f"  AUC-ROC:   {auc_roc if isinstance(auc_roc, str) else f'{auc_roc:.4f}'}")
        
        # Display all hyperparameters
        params = {k.replace('params.', ''): v for k, v in row.items() if k.startswith('params.') and pd.notna(v)}
        if params:
            print("  Hyperparams:")
            for param, value in params.items():
                print(f"    {param}: {value}")
        else:
            print("  Hyperparams: None")

    best_run = runs.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST MODEL (by F1)")
    print(f"{'=' * 80}")
    print(f"Run ID:     {best_run['run_id']}")
    recall = best_run.get('metrics.recall', 'N/A')
    f1 = best_run.get('metrics.f1_score', 'N/A')
    accuracy = best_run.get('metrics.accuracy', 'N/A')
    auc_roc = best_run.get('metrics.roc_auc', 'N/A')
    
    print(f"Recall:     {recall if isinstance(recall, str) else f'{recall:.4f}'}")
    print(f"F1 Score:   {f1 if isinstance(f1, str) else f'{f1:.4f}'}")
    print(f"Accuracy:   {accuracy if isinstance(accuracy, str) else f'{accuracy:.4f}'}")
    print(f"AUC-ROC:    {auc_roc if isinstance(auc_roc, str) else f'{auc_roc:.4f}'}")

    def normalize_value(value):
        if isinstance(value, str):
            if value.lower() == "none":
                return None
            try:
                if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                    return int(value)
                return float(value)
            except ValueError:
                return value
        return value

    VALID_LGB_PARAMS = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "class_weight",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "random_state",
    }

    # Export best LightGBM hyperparameters only
    best_params = {
        key.replace('params.', ''): normalize_value(value)
        for key, value in best_run.items()
        if key.startswith('params.')
        and pd.notna(value)
        and key.replace('params.', '') in VALID_LGB_PARAMS
    }

    if best_params:
        with open("configs/best_model_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
        print(f"\nBest LightGBM hyperparameters exported to configs/best_model_params.yaml")
    else:
        print("\nNo valid LightGBM hyperparameters found to export.")

