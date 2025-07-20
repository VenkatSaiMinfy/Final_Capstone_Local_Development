# src/airflow/scripts/check_drift_runner.py

import os
import json
import sys

import mlflow
from airflow.exceptions import AirflowException
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Ensure project root is on PYTHONPATH to import src modules
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# ─────────────────────────────────────────────
# Load environment variables from .env
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Import helper for loading data from Postgres and drift-check logic
# ─────────────────────────────────────────────
from src.airflow.utils.airflow_loader import load_data
from src.drift.check_drift import check_drift

def run_drift_check():
    """
    1) Configure MLflow tracking URI
    2) Load reference and new preprocessed datasets from Postgres
    3) Run Evidently drift check
    4) Parse drift report JSON to detect any data drift
    Returns:
        bool: True if dataset drift is detected, else False
    """
    # ─────────────────────────────────────────
    # 1) Configure MLflow
    # ─────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise AirflowException("❌ MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(mlflow_uri)

    # ─────────────────────────────────────────
    # 2) Determine table names for reference & new data
    # ─────────────────────────────────────────
    ref_table = os.getenv("DB_TABLE_PREPROCESSED", "preprocessed_train_data")
    new_table = os.getenv("DB_NEW_TABLE_PREPROCESSED", "user_uploaded_preprocessed")

    # ─────────────────────────────────────────
    # 3) Load datasets from Postgres
    # ─────────────────────────────────────────
    try:
        ref_df = load_data(ref_table)
        new_df = load_data(new_table)
    except Exception as e:
        raise AirflowException(f"❌ Failed to load tables '{ref_table}' or '{new_table}': {e}")

    # ─────────────────────────────────────────
    # 4) Execute drift check
    # ─────────────────────────────────────────
    result = check_drift(
        train_df      = ref_df,
        test_df       = new_df,
        dataset_name  = "lead_data_vs_uploaded",
        save_report   = True,
        log_to_mlflow = True
    )

    # ─────────────────────────────────────────
    # 5) Extract raw JSON report string
    # ─────────────────────────────────────────
    raw_json = result.json()
    print("🔍 [DEBUG] result.json() output:")
    print(raw_json)

    # ─────────────────────────────────────────
    # 6) Parse JSON into Python dict
    # ─────────────────────────────────────────
    try:
        report_json = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise AirflowException(f"❌ Invalid JSON from result.json(): {e}")

    # ─────────────────────────────────────────
    # 7) Inspect metrics for dataset_drift flag
    # ─────────────────────────────────────────
    metrics_list = report_json.get("metrics", [])
    print("🔍 [DEBUG] Parsed metrics list:")
    print(json.dumps(metrics_list, indent=2))

    drift_flag = False
    for metric in metrics_list:
        result_dict = metric.get("result", {})
        if "dataset_drift" in result_dict:
            drift_flag = bool(result_dict["dataset_drift"])
            print(f"🔍 [DEBUG] Found dataset_drift={drift_flag} in metric '{metric.get('metric_name')}'")
            break

    return drift_flag

if __name__ == "__main__":
    # For standalone debugging: print drift detection result
    flag = run_drift_check()
    print(f"Dataset drift detected: {flag}")
