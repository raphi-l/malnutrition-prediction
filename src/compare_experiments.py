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

    # Filter to only child runs (search candidates) if nested runs exist
    child_runs = runs[runs['tags.mlflow.parentRunId'].notna()]
    if not child_runs.empty:
        runs = child_runs

    # Sort by recall descending (prioritizing disease detection)
    runs = runs.sort_values('metrics.recall', ascending=False)

    print("Top 5 Runs by Recall Score:")
    print("=" * 80)
    for i, row in runs.head(5).iterrows():
        print(f"\nRun: {row['run_id'][:8]}...")
        print(f"  Recall:    {row['metrics.recall']:.4f}")
        print(f"  F1:        {row['metrics.f1_score']:.4f}")
        print(f"  Accuracy:  {row['metrics.accuracy']:.4f}")
        print(f"  AUC-ROC:   {row['metrics.auc_roc']:.4f}")
        
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
    print("BEST MODEL (by Recall)")
    print(f"{'=' * 80}")
    print(f"Run ID:     {best_run['run_id']}")
    print(f"Recall:     {best_run['metrics.recall']:.4f}")
    print(f"F1 Score:   {best_run['metrics.f1_score']:.4f}")
    print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
    print(f"AUC-ROC:    {best_run['metrics.auc_roc']:.4f}")

    # Export best hyperparameters to YAML for training
    best_params = {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.') and pd.notna(v)}
    if best_params:
        with open("configs/best_model_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
        print(f"\nBest hyperparameters exported to configs/best_model_params.yaml")
    else:
        print("\nNo hyperparameters found to export.")

