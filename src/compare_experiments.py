import sys
import mlflow

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

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.recall DESC"]
    )

    if runs.empty:
        raise SystemExit(
            f"No finished runs found for experiment '{experiment_name}'. "
            "Check that the experiment has completed MLflow runs."
        )

    print("Top 5 Runs by Recall Score:")
    print("=" * 80)
    for i, row in runs.head(5).iterrows():
        print(f"\nRun: {row['run_id'][:8]}...")
        print(f"  F1:       {row['metrics.f1_score']:.4f}")
        print(f"  Accuracy: {row['metrics.accuracy']:.4f}")
        print(f"  AUC-ROC:  {row['metrics.auc_roc']:.4f}")

    best_run = runs.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST MODEL")
    print(f"{'=' * 80}")
    print(f"Run ID:     {best_run['run_id']}")
    print(f"F1 Score:   {best_run['metrics.f1_score']:.4f}")
    print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
    print(f"AUC-ROC:    {best_run['metrics.auc_roc']:.4f}")

    print(f"\n{'=' * 80}")
    print("Average Recall Score by Model Type:")
    print("=" * 80)
