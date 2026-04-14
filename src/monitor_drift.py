import sys
import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

try:
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError
except (IndexError, FileNotFoundError):
    data_path = 'https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_train_sample.csv'

reference = pd.read_csv(data_path)
synth_new_data = pd.read_csv("https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_synth_drift.csv")

print("=" * 60)
print("DRIFT REPORT: Synthetic Quarter 1")
print("=" * 60)

report_q1 = Report(metrics=[DataDriftPreset()])
snapshot_q1 = report_q1.run(reference_data=reference, current_data=synth_new_data)

snapshot_q1.save_html("reports/drift_synthQ1.html")

print("Report saved to reports/drift_synthQ1.html")
print("\nOpen the HTML files in your browser to explore the reports.")

result = snapshot_q1.dict()

def get_drift_summary(reference, current, label):
    """Run drift detection and return a summary dictionary."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    result = snapshot.dict()

    # First metric is DriftedColumnsCount with overall drift info
    drift_count_metric = result["metrics"][0]
    drifted_count = int(drift_count_metric["value"]["count"])
    drift_share = drift_count_metric["value"]["share"]

    # Remaining metrics are per-column ValueDrift
    feature_metrics = result["metrics"][1:]
    total_features = len(feature_metrics)

    summary = {
        "period": label,
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "dataset_drift": drift_share >= 0.5,
    }

    # Extract per-feature drift details
    feature_details = {}
    for metric in feature_metrics:
        column = metric["config"]["column"]
        threshold = metric["config"]["threshold"]
        drift_value = float(metric["value"])
        feature_details[column] = {
            "drifted": drift_value >= threshold,
            "threshold": threshold,
            "drift_score": round(drift_value, 4),
            "method": metric["config"]["method"],
        }

    summary["features"] = feature_details
    return summary