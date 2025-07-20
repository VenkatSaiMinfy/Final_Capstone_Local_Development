import os
import json
import re
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd

# ✅ Compatible with evidently==0.4.40
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Directory to save drift reports
REPORT_DIR = Path("reports/drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_metric_name(name: str) -> str:
    """Sanitize MLflow metric names."""
    return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", name)

def check_drift(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                dataset_name: str = "train_vs_test",
                save_report: bool = True,
                log_to_mlflow: bool = True):
    """
    Runs Evidently DataDrift and DataSummary reports,
    saves outputs, logs metrics to MLflow.
    """

    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not common_cols:
        print(f"⚠️ No common columns found; skipping drift check for {dataset_name}")
        return

    ref = train_df[common_cols].copy()
    cur = test_df[common_cols].copy()

    # Build report with presets
    report = Report(metrics=[
        DataDriftPreset()
    ])
    report.run(reference_data=ref, current_data=cur)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{dataset_name}_{ts}.html"
    json_path = REPORT_DIR / f"drift_{dataset_name}_{ts}.json"

    if save_report:
        report.save_html(str(html_path))
        report.save_json(str(json_path))
        print(f"✅ Saved HTML report to {html_path}")
        print(f"✅ Saved JSON report to {json_path}")

    if log_to_mlflow:
        mlflow.log_artifact(str(html_path), artifact_path=f"drift/{dataset_name}")
        mlflow.log_artifact(str(json_path), artifact_path=f"drift/{dataset_name}")
        print(f"📦 MLflow: Logged artifacts under drift/{dataset_name}")

        # Load metrics from JSON
        with open(json_path, "r") as f:
            report_json = json.load(f)

        for m in report_json.get("metrics", []):
            val = m.get("value", None)
            metric_id = m.get("metric_id") or m.get("metric") or ""

            if val is None and isinstance(m.get("result"), dict):
                for k in (
                    "drift_score", "mean", "mean_reference", "mean_current",
                    "number_of_rows", "number_of_columns",
                    "number_of_drifted_columns", "share_of_drifted_columns"
                ):
                    if k in m["result"]:
                        val = m["result"][k]
                        metric_id = f"{metric_id}_{k}"
                        break

            if isinstance(val, (int, float)):
                safe_name = f"{dataset_name}__{_sanitize_metric_name(metric_id)}"
                mlflow.log_metric(safe_name, float(val))
                print(f"📊 MLflow metric: {safe_name} = {val}")

        print(f"✅ Logged numeric metrics for `{dataset_name}`")

    return report
