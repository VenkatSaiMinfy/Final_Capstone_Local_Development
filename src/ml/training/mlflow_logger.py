import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ───────────────────────────────────────────────────────────────
# 📦 MLflow Logger Utility
# Logs a model along with its metrics, parameters, and input signature
# ───────────────────────────────────────────────────────────────
def mlflow_logger(model_name, model, metrics, params, X_sample=None):
    """
    Logs a scikit-learn model to MLflow with parameters, metrics, and optional input signature.

    Parameters:
    - model_name (str): Name for the MLflow run and artifact.
    - model (sklearn model): Trained scikit-learn model.
    - metrics (dict): Dictionary of evaluation metrics.
    - params (dict): Dictionary of training parameters.
    - X_sample (DataFrame or ndarray, optional): Sample input data to infer signature and input example.
    """
    
    # ─────────────────────────────────────────────────────
    # Handle any existing active run (to avoid MLflow errors)
    # ─────────────────────────────────────────────────────
    if mlflow.active_run():
        print(f"⚠️ Active MLflow run detected. Ending it before starting a new one.")
        mlflow.end_run()

    try:
        print(f"\n🕒 Starting MLflow run for model: '{model_name}'")
        mlflow.start_run(run_name=model_name)

        # ───────────────────────────────────────────────
        # Log hyperparameters and evaluation metrics
        # ───────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # ───────────────────────────────────────────────
        # Try to log input signature and example for deployment
        # ───────────────────────────────────────────────
        signature = None
        input_example = None
        if X_sample is not None:
            try:
                y_pred = model.predict(X_sample)
                signature = infer_signature(X_sample, y_pred)
                input_example = X_sample[:5]  # Only a few rows to keep it light
            except Exception as e:
                print(f"⚠️ Could not infer signature: {e}")

        # ───────────────────────────────────────────────
        # Log the trained model into MLflow artifacts
        # ───────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            input_example=input_example,
            signature=signature
        )
        print(f"[MLflow ✅] Model logged under: '{model_name}'")

    except Exception as e:
        print(f"❌ Error while logging model to MLflow: {e}")

    finally:
        # Always clean up and end the MLflow run
        if mlflow.active_run():
            mlflow.end_run()
