# src/drift/check_drift.py

import os
import json
import re
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
REPORT_DIR = Path("reports/drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _sanitize_metric_name(name: str) -> str:
    """
    Replace characters not allowed in MLflow metric names with underscores.
    Allowed: A-Z, a-z, 0-9, _, -, ., space, :, /
    """
    return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", name)

# ─────────────────────────────────────────────────────────────
# Main Drift Check Function
# ─────────────────────────────────────────────────────────────
def check_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str = "train_vs_test",
    save_report: bool = True,
    log_to_mlflow: bool = True
):
    """
    Runs Evidently's DataDriftPreset + DataSummaryPreset to compare datasets.

    Args:
        train_df (pd.DataFrame): Reference (training) dataset.
        test_df (pd.DataFrame): Current (incoming/test) dataset.
        dataset_name (str): Name used for logging and filenames.
        save_report (bool): If True, saves HTML and JSON reports to disk.
        log_to_mlflow (bool): If True, logs artifacts and metrics to MLflow.

    Returns:
        evidently.Report: The generated drift report.
    """
    # ─────────────────────────────────────────────────────────
    # Step 1: Find common columns between datasets
    # ─────────────────────────────────────────────────────────
    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping drift check.")
        return

    ref = train_df[common_cols].copy()
    cur = test_df[common_cols].copy()

    # ─────────────────────────────────────────────────────────
    # Step 2: Run Drift and Summary Report
    # ─────────────────────────────────────────────────────────
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{dataset_name}_{timestamp}.html"
    json_path = REPORT_DIR / f"drift_{dataset_name}_{timestamp}.json"

    # ─────────────────────────────────────────────────────────
    # Step 3: Save Reports
    # ─────────────────────────────────────────────────────────
    if save_report:
        result.save_html(str(html_path))
        result.save_json(str(json_path))
        print(f"✅ Drift HTML saved to {html_path}")
        print(f"📦 Drift JSON saved to {json_path}")

    # ─────────────────────────────────────────────────────────
    # Step 4: Log to MLflow
    # ─────────────────────────────────────────────────────────
    if log_to_mlflow:
        mlflow.log_artifact(str(html_path), artifact_path=f"drift/{dataset_name}")
        mlflow.log_artifact(str(json_path), artifact_path=f"drift/{dataset_name}")
        print(f"✅ Logged artifacts under drift/{dataset_name}")

        # Parse and log selected numeric metrics
        report_json = json.loads(result.json())
        for metric in report_json.get("metrics", []):
            value = metric.get("value")
            metric_id = metric.get("metric_id") or metric.get("metric", "")

            # Handle nested `result` values
            if value is None and isinstance(metric.get("result"), dict):
                for key in [
                    "drift_score", "mean", "mean_reference", "mean_current",
                    "number_of_rows", "number_of_columns",
                    "number_of_drifted_columns", "share_of_drifted_columns"
                ]:
                    if key in metric["result"]:
                        value = metric["result"][key]
                        metric_id = f"{metric_id}_{key}"
                        break

            if isinstance(value, (int, float)):
                safe_metric_name = f"{dataset_name}__{_sanitize_metric_name(metric_id)}"
                mlflow.log_metric(safe_metric_name, float(value))
                print(f"🔢 {safe_metric_name} = {value}")

        print(f"✅ Logged all numeric drift & summary metrics for `{dataset_name}`")

    return result
