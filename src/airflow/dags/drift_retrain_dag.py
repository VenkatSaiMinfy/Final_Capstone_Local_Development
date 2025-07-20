# src/airflow/dags/drift_retrain_dag.py

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ─────────────────────────────────────────────
# Dynamically add project root to PYTHONPATH so `src` modules can be imported
# ─────────────────────────────────────────────
CURRENT_FILE_PATH = os.path.abspath(__file__)
while True:
    CURRENT_FILE_PATH = os.path.dirname(CURRENT_FILE_PATH)
    # Stop when we find the directory containing `src/`
    if os.path.exists(os.path.join(CURRENT_FILE_PATH, "src")):
        break

if CURRENT_FILE_PATH not in sys.path:
    sys.path.insert(0, CURRENT_FILE_PATH)

# ─────────────────────────────────────────────
# Import custom Airflow task callables
# ─────────────────────────────────────────────
from src.airflow.scripts.trigger_upload_monitor import has_new_upload
from src.airflow.scripts.check_drift_runner import run_drift_check
from src.airflow.scripts.retrain_runner import run_retrain

# ─────────────────────────────────────────────
# Default arguments applied to all tasks in this DAG
# ─────────────────────────────────────────────
default_args = {
    "owner": "VenkatSai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ─────────────────────────────────────────────
# Define the DAG: hourly schedule, no catchup, tags for organization
# ─────────────────────────────────────────────
with DAG(
    dag_id="drift_and_retrain",
    default_args=default_args,
    description="Detect data drift and retrain model when needed",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["lead_scoring", "drift"],
) as dag:

    # ─────────────────────────────────────────
    # Task: perform data drift check
    # Uses `run_drift_check`, returns True/False via XCom
    # ─────────────────────────────────────────
    check_drift = PythonOperator(
        task_id="perform_drift_check",
        python_callable=run_drift_check,
    )

    # ─────────────────────────────────────────
    # Task: decide whether to retrain
    # Branches based on XCom result from `perform_drift_check`
    # ─────────────────────────────────────────
    branch_retrain = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=lambda drift: "retrain_model" if drift else "end",
        op_args=["{{ ti.xcom_pull(task_ids='perform_drift_check') }}"],
    )

    # ─────────────────────────────────────────
    # Task: retrain the model if drift detected
    # ─────────────────────────────────────────
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retrain,
    )

    # ─────────────────────────────────────────
    # Task: end of workflow (no operation)
    # ─────────────────────────────────────────
    end = EmptyOperator(task_id="end")

    # ─────────────────────────────────────────
    # Set task dependencies:
    # 1) check_drift → branch_retrain
    # 2) branch_retrain → retrain_model or end
    # 3) retrain_model → end
    # ─────────────────────────────────────────
    check_drift >> branch_retrain
    branch_retrain >> [retrain, end]
    retrain >> end
