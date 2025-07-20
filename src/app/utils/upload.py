#!/usr/bin/env python3
"""
src/app/utils/upload.py

Handles loading a user-uploaded CSV file into a PostgreSQL table.
"""

import os
import sys

# ─────────────────────────────────────────────
# Add project root to PYTHONPATH for local imports
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# ─────────────────────────────────────────────
# Import shared CSV-to-Postgres utility
# ─────────────────────────────────────────────
from src.ml.data_loader.data_loader import load_csv_to_postgres

def handle_csv_upload(filepath: str, table_name: str):
    """
    Load the CSV file at `filepath` into the specified Postgres table.
    
    Args:
        filepath (str): Path to the CSV file to upload.
        table_name (str): Name of the target Postgres table.
    """
    # Delegate CSV loading to the shared utility function,
    # replacing any existing data in `table_name`
    load_csv_to_postgres(
        csv_path=filepath,
        table_name=table_name,
        if_exists="replace"
    )
