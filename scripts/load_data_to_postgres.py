# scripts/load_data_to_postgres.py

import sys
import os
import pandas as pd

# ─────────────────────────────────────────────
# Ensure project root is on PYTHONPATH so local modules can be imported
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────
# Import database engine utility and CSV loader function
# ─────────────────────────────────────────────
from src.db.db_utils import get_db_engine
from src.ml.data_loader.data_loader import load_csv_to_postgres

# ─────────────────────────────────────────────
# Define source CSV path and target Postgres table name
# ─────────────────────────────────────────────
csv_path = "data/lead_scoring.csv"
table_name = os.getenv("DB_TABLE")

# ─────────────────────────────────────────────
# Main execution block: load the CSV into the specified table
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_csv_to_postgres(csv_path, table_name)
