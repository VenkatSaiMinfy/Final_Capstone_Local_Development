# src/airflow/utils/airflow_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine
from airflow.exceptions import AirflowException

def load_data(table_name: str) -> pd.DataFrame:
    """
    Load an entire PostgreSQL table into a pandas DataFrame.
    
    Expects the following environment variables to be set:
      • DB_HOST:     hostname or IP of the Postgres server
      • DB_PORT:     port number (defaults to '5432' if unset)
      • DB_NAME:     database name
      • DB_USER:     username
      • DB_PASSWORD: password
    
    Args:
        table_name (str): Name of the table to load.
        
    Returns:
        pd.DataFrame: Contents of the table.
    
    Raises:
        AirflowException: If credentials are missing or the load fails.
    """
    # ─────────────────────────────────────────
    # 1) Read database credentials from environment
    # ─────────────────────────────────────────
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")

    # ─────────────────────────────────────────
    # 2) Verify all required credentials are present
    # ─────────────────────────────────────────
    if not all([host, port, db, user, pwd]):
        raise AirflowException(
            "Postgres credentials not fully set. "
            "Please define DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD."
        )

    # ─────────────────────────────────────────
    # 3) Construct SQLAlchemy connection string & engine
    # ─────────────────────────────────────────
    conn_str = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    engine = create_engine(conn_str)

    try:
        # ─────────────────────────────────────────
        # 4) Load table into DataFrame
        # ─────────────────────────────────────────
        df = pd.read_sql_table(table_name, engine)
    except Exception as e:
        # ─────────────────────────────────────────
        # 5) Wrap any failure in an AirflowException
        # ─────────────────────────────────────────
        raise AirflowException(f"Failed to load table '{table_name}': {e}")
    finally:
        # ─────────────────────────────────────────
        # 6) Dispose of the engine to close connections
        # ─────────────────────────────────────────
        engine.dispose()

    return df
