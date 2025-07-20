import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
import mlflow

# Allow relative imports when running as __main__
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.eda.profiler import generate_eda_report
from src.ml.data_loader.data_loader import load_data_from_postgres, save_dataframe_to_postgres
from src.ml.pipeline.preprocessing import clean_columns, get_full_pipeline
from src.ml.pipeline.feature_selection import apply_feature_selection
from src.ml.pipeline.feature_selector import FeatureSelector
from src.ml.registry.model_registry import register_and_promote


def print_time(step: str, t0: datetime) -> datetime:
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[⏱️] {step} finished in {elapsed:.2f}s")
    return datetime.now()


def run_pipeline(
    table_name: str = "lead_data",
    target_col: str = "Converted",
    save: bool = True,
    register: bool = False,
    return_pipeline: bool = False
):
    """
    Runs the full preprocessing pipeline: load, clean, transform, feature selection, and save.

    Args:
        table_name (str): Name of the table to read from Redshift/Postgres.
        target_col (str): Name of the target column.
        save (bool): Whether to save transformed data and pipeline.
        register (bool): Whether to log pipeline in MLflow.
        return_pipeline (bool): Whether to return final pipeline and selected data.

    Returns:
        Tuple[X_selected, y, final_pipeline] if return_pipeline is True,
        else just (X_selected, y)
    """
    t0 = datetime.now()

    # 1. Load & clean data
    df = load_data_from_postgres(table_name)
    print(f"[INFO] Loaded data from '{table_name}' with shape: {df.shape}")
    generate_eda_report(df)
    df = clean_columns(df)
    t0 = print_time("Data load & clean", t0)

    # 2. Split target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Define feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    t0 = print_time("Feature type identification", t0)

    # 4. Build & fit full pipeline
    full_pipeline = get_full_pipeline(numeric_features, categorical_features)
    X_transformed = full_pipeline.fit_transform(X, y)
    t0 = print_time("Pipeline fit & transform", t0)

    # 5. Get transformed feature names
    try:
        feature_names = full_pipeline.named_steps["preprocessing"].get_feature_names_out()
    except (AttributeError, KeyError):
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    t0 = print_time("Extracting feature names", t0)

    # 6. Apply RFE feature selection
    X_selected, selected_indices = apply_feature_selection(X_transformed, y)
    t0 = print_time("Feature selection with RFE", t0)

    # 7. Construct final pipeline for inference
    final_pipeline = Pipeline([
        ("feature_engineering", full_pipeline.named_steps["feature_engineering"]),
        ("preprocessing",       full_pipeline.named_steps["preprocessing"]),
        ("feature_selection",   FeatureSelector(selected_features=selected_indices)),
    ])
    final_pipeline.fit(X, y)
    t0 = print_time("Final pipeline construction", t0)

    # 8. Save pipeline and transformed data
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_pipeline, "models/full_pipeline.pkl", compress=3)
        print("✅ Saved pipeline to models/full_pipeline.pkl")

        selected_names = [feature_names[i] for i in selected_indices]
        df_pre = pd.DataFrame(X_selected, columns=selected_names)
        save_dataframe_to_postgres(df_pre, table_name="preprocessed_train_data")
        print(f"✅ Saved selected features to 'preprocessed_train_data' with columns: {selected_names}")
        t0 = print_time("Artifact saving", t0)

    # 9. MLflow registration (optional)
    if register:
        try:
            register_and_promote(
                registry_name="LeadScoringPreprocessor",
                model_object=final_pipeline,
                is_pipeline=True
            )
            print("✅ Pipeline registered in MLflow as 'LeadScoringPreprocessor'")
        except Exception as e:
            print(f"❌ Pipeline registration failed: {e}")
        t0 = print_time("MLflow registration", t0)

    # 10. Return pipeline (optional)
    if return_pipeline:
        return X_selected, y, final_pipeline
    return X_selected, y


if __name__ == "__main__":
    run_pipeline(save=True, register=True)
