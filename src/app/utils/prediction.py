#!/usr/bin/env python3
"""
predict.py

Provides single-record and batch prediction functions for the Lead Scoring model.
Loads the preprocessing pipeline and trained classifier from the MLflow registry,
applies transforms, saves preprocessed features to Postgres (for batch), and returns predictions.
"""

import os
import sys
from typing import Union, List

import pandas as pd
import numpy as np
import mlflow.sklearn

# ─────────────────────────────────────────────
# Add project root to PYTHONPATH for local imports
# ─────────────────────────────────────────────
here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ml.data_loader.data_loader import save_dataframe_to_postgres

# ─────────────────────────────────────────────
# Constants: MLflow model registry names and target stage
# ─────────────────────────────────────────────
PREPROCESSOR_NAME = "LeadScoringPreprocessor"
MODEL_NAME = "LeadScoringBestModel"
STAGE = "Production"

# ─────────────────────────────────────────────
# Load preprocessing pipeline & classifier from MLflow registry
# ─────────────────────────────────────────────
try:
    # Full pipeline: feature_engineering → preprocessing → feature_selection
    preprocessor_uri = f"models:/{PREPROCESSOR_NAME}/{STAGE}"
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)

    # Classifier for predict_proba
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"✅ Loaded preprocessor and model (stage={STAGE})")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load preprocessor or model: {e}")


def predict_lead(input_dict: dict) -> Union[float, dict]:
    """
    Predict conversion probability for a single lead.
    
    Args:
        input_dict (dict): Raw feature values for one lead.
    
    Returns:
        float: Probability of conversion, or dict with "error" on failure.
    """
    try:
        # 1) Wrap input in DataFrame
        df = pd.DataFrame([input_dict])

        # 2) Apply full preprocessing pipeline
        X_proc = preprocessor.transform(df)

        # 3) Get raw probabilities
        raw = model.predict_proba(X_proc)
        arr = np.asarray(raw)

        # 4) Handle output shapes
        if arr.ndim == 1:               # e.g., regressors or single-proba
            return float(arr[0])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])    # probability of positive class
        if arr.ndim == 2 and arr.shape[1] == 1:
            return float(arr[0, 0])

        return {"error": f"Unexpected output shape: {arr.shape}"}
    except Exception as e:
        return {"error": str(e)}


def predict_batch(df: pd.DataFrame, save: bool = True) -> Union[List[int], dict]:
    """
    Predict conversion for multiple leads and optionally save preprocessed features.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame (unprocessed features).
        save (bool): If True, save preprocessed features to Postgres.
    
    Returns:
        List[int]: Binary predictions (0/1) list or dict with "error" on failure.
    """
    try:
        # 1) Transform raw inputs through full pipeline
        X_proc = preprocessor.transform(df)  # shape: (n_rows, n_selected_features)

        if save:
            # 2) Retrieve full encoded feature names from preprocessing step
            preprocessing = preprocessor.named_steps["preprocessing"]
            all_feature_names = preprocessing.get_feature_names_out()  # e.g., 192 features

            # 3) Retrieve indices of features selected by RFE
            selector = preprocessor.named_steps["feature_selection"]
            selected_indices = selector.selected_features  # e.g., 50 ints

            # 4) Map indices to final feature names
            feature_names = [all_feature_names[i] for i in selected_indices]

            # 5) Ensure names match transformed data shape
            assert len(feature_names) == X_proc.shape[1], (
                f"Expected {X_proc.shape[1]} names, got {len(feature_names)}"
            )

            # 6) Build DataFrame of preprocessed features with real column names
            df_pre = pd.DataFrame(X_proc, columns=feature_names)

            # 7) Save to Postgres table for monitoring or drift checks
            save_dataframe_to_postgres(
                df_pre,
                table_name="user_uploaded_preprocessed",
                if_exists="append"
            )
            print(f"✅ Saved preprocessed batch to 'user_uploaded_preprocessed' with columns: {feature_names}")

        # 8) Generate predictions from classifier
        raw = model.predict_proba(X_proc)
        arr = np.asarray(raw)
        if arr.ndim == 1:
            preds = [int(x > 0.5) for x in arr]
        else:
            preds = [int(x > 0.5) for x in arr[:, 1]]

        return preds
    except Exception as e:
        print("❌ [ERROR] in predict_batch:", e)
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Debug: run sample predictions when executed directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Create a dummy input dict with zeros for every expected raw feature
    sample = {col: 0 for col in preprocessor.feature_names_in_}
    print("Single prediction:", predict_lead(sample))

    batch_df = pd.DataFrame([sample, sample])
    print("Batch predictions:", predict_batch(batch_df, save=False))
