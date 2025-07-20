# ────────────────────────────────────────────────────────────────
# data_loader.py – Load and save data between CSV/PostgreSQL
# ────────────────────────────────────────────────────────────────

import os
import pandas as pd
from src.db.db_utils import get_db_engine  # ✅ Shared DB engine utility


# ─────────────────────────────────────────────
# Load data from PostgreSQL table
# ─────────────────────────────────────────────
def load_data_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Load data from a PostgreSQL table into a pandas DataFrame.

    Args:
        table_name (str): Name of the table to query.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load data from '{table_name}': {e}")


# ─────────────────────────────────────────────
# Load data from CSV to PostgreSQL
# ─────────────────────────────────────────────
def load_csv_to_postgres(csv_path: str, table_name: str, if_exists: str = "replace"):
    """
    Load a CSV file into a PostgreSQL table.

    Args:
        csv_path (str): Path to the CSV file.
        table_name (str): Target table name.
        if_exists (str): What to do if table exists: 'replace', 'append', or 'fail'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        engine = get_db_engine()
        df.to_sql(table_name, engine, index=False, if_exists=if_exists)
        print(f"✅ CSV data loaded into table '{table_name}' (if_exists='{if_exists}')")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load CSV to PostgreSQL: {e}")


# ─────────────────────────────────────────────
# Save DataFrame to PostgreSQL
# ─────────────────────────────────────────────
def save_dataframe_to_postgres(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    """
    Save a pandas DataFrame to a PostgreSQL table.

    Args:
        df (pd.DataFrame): Data to save.
        table_name (str): Target table name.
        if_exists (str): What to do if table exists: 'replace', 'append', or 'fail'.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The DataFrame is empty and cannot be saved.")

    try:
        engine = get_db_engine()
        df.to_sql(table_name, engine, index=False, if_exists=if_exists)
        print(f"✅ DataFrame saved to PostgreSQL table '{table_name}' (if_exists='{if_exists}')")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save DataFrame to PostgreSQL: {e}")
