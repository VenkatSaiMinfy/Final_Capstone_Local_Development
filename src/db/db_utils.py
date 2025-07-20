# src/db/db_utils.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load database credentials from .env into environment
# ─────────────────────────────────────────────
load_dotenv()

def get_db_engine():
    """
    Create and return a SQLAlchemy engine for connecting to PostgreSQL.
    
    Expects the following environment variables to be set:
      • DB_USER     - database username
      • DB_PASSWORD - database password
      • DB_HOST     - hostname or IP of the database server
      • DB_PORT     - port number (e.g., "5432")
      • DB_NAME     - name of the target database
    
    Raises:
        EnvironmentError: If any credential is missing.
    
    Returns:
        sqlalchemy.Engine: Engine instance for database connections.
    """
    # ─────────────────────────────────────────
    # 1) Read credentials from environment
    # ─────────────────────────────────────────
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # ─────────────────────────────────────────
    # 2) Validate that all required credentials are present
    # ─────────────────────────────────────────
    if not all([db_user, db_pass, db_host, db_port, db_name]):
        raise EnvironmentError(
            "Database credentials are not fully set in .env. "
            "Please define DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, and DB_NAME."
        )

    # ─────────────────────────────────────────
    # 3) Construct the database connection URL
    # ─────────────────────────────────────────
    connection_url = (
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )

    # ─────────────────────────────────────────
    # 4) Create and return the SQLAlchemy engine
    # ─────────────────────────────────────────
    return create_engine(connection_url)
