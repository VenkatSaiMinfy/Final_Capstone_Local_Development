import pandas as pd

# ─────────────────────────────────────────────────────────────
# ✅ Schema Validator
# ─────────────────────────────────────────────────────────────

def validate_input_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validates whether all expected columns exist in the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to validate
        expected_columns (list[str]): List of expected column names
    
    Returns:
        bool: True if validation passes, otherwise raises ValueError

    Raises:
        ValueError: If any expected columns are missing
    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Schema validation failed. Missing columns: {missing}")
    
    return True
