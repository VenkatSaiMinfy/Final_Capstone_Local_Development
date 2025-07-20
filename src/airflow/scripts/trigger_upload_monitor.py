# src/airflow/scripts/trigger_upload_monitor.py

import os
from datetime import datetime

import pandas as pd
from airflow.exceptions import AirflowSkipException
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables from .env (if not already loaded)
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Path to file that tracks the last upload-check timestamp
# ─────────────────────────────────────────────
LAST_RUN_PATH = os.getenv("LAST_RUN_FILE", "/tmp/last_upload_check.txt")

# ─────────────────────────────────────────────
# Import data-loading utility after PYTHONPATH is set by Airflow
# ─────────────────────────────────────────────
from src.airflow.utils.airflow_loader import load_data

def has_new_upload():
    """
    Checks for new rows in the 'user_uploaded_preprocessed' table since the last run.
    Relies on an 'uploaded_at' timestamp column.
    Returns:
        True if new data exists (and updates LAST_RUN_PATH).
    Raises:
        AirflowSkipException if no new data or column missing.
    """
    # ─────────────────────────────────────────
    # 1️⃣ Determine which table to monitor
    # ─────────────────────────────────────────
    table_name = os.getenv("DB_NEW_TABLE_PREPROCESSED", "user_uploaded_preprocessed")
    print(f"[INFO] Checking for new uploads in table: {table_name}")

    # ─────────────────────────────────────────
    # 2️⃣ Load the table into a DataFrame
    # ─────────────────────────────────────────
    df = load_data(table_name)

    # ─────────────────────────────────────────
    # 3️⃣ Ensure the 'uploaded_at' column exists
    # ─────────────────────────────────────────
    if "uploaded_at" not in df.columns:
        raise AirflowSkipException(
            f"[SKIP] Table '{table_name}' missing required 'uploaded_at' column."
        )

    # ─────────────────────────────────────────
    # 4️⃣ Parse timestamps and find the latest one
    # ─────────────────────────────────────────
    df["uploaded_at"] = pd.to_datetime(df["uploaded_at"])
    most_recent = df["uploaded_at"].max()
    print(f"[INFO] Latest upload timestamp in DB: {most_recent}")

    # ─────────────────────────────────────────
    # 5️⃣ Read the timestamp of the last successful check
    # ─────────────────────────────────────────
    try:
        with open(LAST_RUN_PATH, "r") as f:
            last_run = pd.to_datetime(f.read().strip())
            print(f"[INFO] Last checked timestamp: {last_run}")
    except FileNotFoundError:
        # First run ever, proceed as if new data exists
        last_run = datetime.min
        print("[INFO] First run - no previous timestamp found.")

    # ─────────────────────────────────────────
    # 6️⃣ Compare timestamps and decide
    # ─────────────────────────────────────────
    if most_recent > last_run:
        print("[INFO] New data detected - proceeding to drift check.")
        # Update the last-run file
        with open(LAST_RUN_PATH, "w") as f:
            f.write(most_recent.isoformat())
        return True

    # ─────────────────────────────────────────
    # 7️⃣ No new uploads since last run → skip downstream tasks
    # ─────────────────────────────────────────
    print("[SKIP] No new uploads detected since last check.")
    raise AirflowSkipException("No new data uploaded since last check.")
