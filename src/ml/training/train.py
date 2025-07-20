# src/ml/training/train.py

import os
import sys
from datetime import datetime
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Path setup for relative imports + load environment variables
# ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
load_dotenv()

# ─────────────────────────────────────────────
# Internal module imports
# ─────────────────────────────────────────────
from src.drift.check_drift import check_drift
from src.ml.pipeline.pipeline_runner import run_pipeline
from src.ml.training.train_utils import get_models_with_params, train_and_log_model
from src.ml.registry.model_registry import register_and_promote


def train_all_models():
    """
    Executes full pipeline: preprocessing, model training, drift detection, 
    MLflow logging, and best model registration.
    """

    # ─────────────────────────────────────────────
    # 1. Preprocess & Feature Selection Pipeline
    # ─────────────────────────────────────────────
    X_sel, y, final_pipeline = run_pipeline(
        save=True,
        register=True,
        return_pipeline=True
    )

    # ─────────────────────────────────────────────
    # 2. Extract final feature names from pipeline
    # ─────────────────────────────────────────────
    preproc = final_pipeline.named_steps["preprocessing"]
    all_names = preproc.get_feature_names_out()                       # All columns post-transform
    sel_idxs = final_pipeline.named_steps["feature_selection"].selected_features
    feat_names = all_names[sel_idxs]                                  # Final selected feature names

    # ─────────────────────────────────────────────
    # 3. Convert selected features to DataFrame
    # ─────────────────────────────────────────────
    df = pd.DataFrame(X_sel, columns=feat_names)

    # ─────────────────────────────────────────────
    # 4. Train/Test Split
    # ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2,
        random_state=42, stratify=y
    )

    # ─────────────────────────────────────────────
    # 5. Configure MLflow experiment
    # ─────────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))
    mlflow.set_experiment("Lead Scoring Model")

    best_f1 = -1.0
    best_info = (None, None)  # (model_name, run_id)

    # ─────────────────────────────────────────────
    # 6. MLflow Parent Run: Track all child runs
    # ─────────────────────────────────────────────
    with mlflow.start_run(run_name="All_Model_Training_Run") as parent:
        print(f"[INFO] Parent run ID: {parent.info.run_id}")

        # 🧪 Check data drift between train & test
        check_drift(X_train, X_test, "train_vs_test", save_report=True, log_to_mlflow=True)

        # 🔁 Train and log all models
        for name, (model, params) in get_models_with_params().items():
            mname, run_id, f1 = train_and_log_model(
                name=name,
                model=model,
                param_grid=params,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feat_names
            )

            # 🥇 Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_info = (mname, run_id)

        # ✅ Register best model to MLflow Model Registry
        best_name, best_run = best_info
        if best_name:
            print(f"\n🏆 Best model: {best_name} (F1={best_f1:.4f})")
            uri = f"runs:/{best_run}/{best_name}"
            register_and_promote(
                registry_name="LeadScoringBestModel",
                run_id=best_run,
                model_uri=uri,
                is_pipeline=False
            )
        else:
            print("❌ No successful model runs to register.")

    # ⛔ Parent run ends automatically here


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all_models()
