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

report_month1 = Report(metrics=[DataDriftPreset()])
snapshot_month1 = report_month1.run(reference_data=reference, current_data=synth_new_data)

snapshot_month1.save_html("reports/drift_synthQ1.html")
print("Report saved to reports/drift_synthQ1.html")


print("\nOpen the HTML files in your browser to explore the reports.")