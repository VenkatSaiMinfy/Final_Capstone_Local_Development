# src/eda/profiler.py

import os
import sys
import mlflow
from ydata_profiling import ProfileReport

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.data_loader.data_loader import load_data_from_postgres
from src.db.db_utils import get_db_engine

DATA_PATH = os.path.join("data", "lead_scoring.csv")

def generate_eda_report(df, log_to_mlflow=True):
    output_path = os.path.join("data", "eda_report.html")

    profile = ProfileReport(df, title="EDA Report - Lead Scoring", minimal=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile.to_file(output_path)

    print(f"📊 EDA report saved to: {output_path}")

    # ✅ Log to MLflow if inside a run
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_artifact(output_path, artifact_path="eda")
        print("✅ EDA report logged to MLflow under 'eda/'")


if __name__ == "__main__":
    data = load_data_from_postgres(os.getenv("DB_TABLE"))
    generate_eda_report(data)
