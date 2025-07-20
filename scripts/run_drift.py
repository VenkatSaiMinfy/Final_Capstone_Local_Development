# scripts/run_drift.py

import os
import sys
from datetime import datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Ensure project root is on PYTHONPATH to import src modules
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────
# Import data loading, preprocessing, and drift-check utilities
# ─────────────────────────────────────────────
from src.ml.data_loader.data_loader import load_data_from_postgres
from src.ml.pipeline.preprocessing import clean_columns
from src.ml.pipeline.feature_engineering import feature_engineering
from src.drift.check_drift import check_drift

# ─────────────────────────────────────────────
# Main entry point: load data, preprocess, split, and run drift checks
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1️⃣ Load raw data from Postgres and apply cleaning & feature engineering
    df = load_data_from_postgres("lead_data")
    df = clean_columns(df)
    df = feature_engineering(df)

    # Split into train/test for drift comparison
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # ─────────────────────────────────────────
    # 2️⃣ Prepare MLflow experiment for drift checks
    # ─────────────────────────────────────────
    exp_name = "Lead Scoring Drift Checks"
    client = MlflowClient()

    # Create or restore the experiment if it doesn't exist or was deleted
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name)
        print(f"ℹ️ Created experiment '{exp_name}'")
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        print(f"ℹ️ Restored deleted experiment '{exp_name}'")

    # Set the active MLflow experiment
    mlflow.set_experiment(exp_name)

    # ─────────────────────────────────────────
    # 3️⃣ Execute and log the drift check run
    # ─────────────────────────────────────────
    with mlflow.start_run(run_name="train_test_drift_check"):
        print("📊 Checking drift between train and test datasets...")
        check_drift(
            train_df=train_df,
            test_df=test_df,
            dataset_name="train_vs_test",
            save_report=True,
            log_to_mlflow=True
        )
