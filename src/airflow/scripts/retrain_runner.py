# src/airflow/scripts/retrain_runner.py

import os
from src.ml.pipeline.pipeline_runner import run_pipeline

def run_retrain():
    """
    Retrains the lead scoring model using the full reference dataset,
    saves artifacts, and registers/promotes the pipeline in MLflow.
    
    Returns:
        bool: True when retraining completes successfully.
    """
    # ────────────────────────────────────────────
    # 1️⃣ Ensure MLflow tracking URI is configured
    # ────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise EnvironmentError("❌ MLFLOW_TRACKING_URI environment variable is not set")
    # Set it explicitly in case Airflow's environment needs it at runtime
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    # ────────────────────────────────────────────
    # 2️⃣ Execute the full training pipeline
    #    - Loads data from `REFERENCE_TABLE` (default 'lead_data')
    #    - Cleans, engineers features, selects, trains, and logs to MLflow
    #    - Saves artifacts locally and to Postgres (if configured)
    #    - Registers/promotes the pipeline as 'LeadScoringPreprocessor'
    # ────────────────────────────────────────────
    run_pipeline(
        table_name=os.getenv("REFERENCE_TABLE", "lead_data"),
        target_col="Converted",
        save=True,
        register=True
    )

    # ────────────────────────────────────────────
    # 3️⃣ Return True to signal successful retraining
    #    (used by Airflow's BranchPythonOperator)
    # ────────────────────────────────────────────
    return True
