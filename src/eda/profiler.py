# ────────────────────────────────────────────────────────────────
# profiler.py – Generates EDA report using ydata-profiling and logs to MLflow
# ────────────────────────────────────────────────────────────────

import os
import sys
import mlflow
from ydata_profiling import ProfileReport

# ─────────────────────────────────────────────
# Add project root to PYTHONPATH (for imports)
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─────────────────────────────────────────────
# Local Imports
# ─────────────────────────────────────────────
from src.ml.data_loader.data_loader import load_data_from_postgres
from src.db.db_utils import get_db_engine  # Optional if used elsewhere

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_PATH = os.path.join("data", "lead_scoring.csv")
EDA_OUTPUT_PATH = os.path.join("data", "eda_report.html")


# ─────────────────────────────────────────────
# Function: Generate and log EDA Report
# ─────────────────────────────────────────────
def generate_eda_report(df, log_to_mlflow=True):
    """
    Generate EDA report using ydata-profiling, save as HTML,
    and optionally log to MLflow if a run is active.

    Args:
        df (pd.DataFrame): DataFrame to profile.
        log_to_mlflow (bool): Whether to log the report to MLflow.
    """
    os.makedirs(os.path.dirname(EDA_OUTPUT_PATH), exist_ok=True)

    # Create report
    profile = ProfileReport(df, title="EDA Report - Lead Scoring", minimal=True)
    profile.to_file(EDA_OUTPUT_PATH)

    print(f"📊 EDA report saved to: {EDA_OUTPUT_PATH}")

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_artifact(EDA_OUTPUT_PATH, artifact_path="eda")
        print("✅ EDA report logged to MLflow under 'eda/'")


# ─────────────────────────────────────────────
# Entry point: Run EDA on table from DB
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load data from Redshift/Postgres using DB_TABLE env var
    table_name = os.getenv("DB_TABLE")
    if not table_name:
        raise ValueError("❌ Environment variable DB_TABLE not set.")

    data = load_data_from_postgres(table_name)
    generate_eda_report(data)
