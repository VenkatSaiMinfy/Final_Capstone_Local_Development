import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 🏷️ Ensure Model Registry Entry Exists
# ─────────────────────────────────────────────────────────────
def _ensure_model_registered(client, registry_name: str):
    """
    Ensures a model with the given name exists in the MLflow model registry.
    Creates it if not already present.
    """
    try:
        client.get_registered_model(registry_name)
    except MlflowException as e:
        if "not found" in str(e).lower():
            print(f"ℹ️ Registered model '{registry_name}' not found. Creating it...")
            client.create_registered_model(registry_name)
        else:
            raise

# ─────────────────────────────────────────────────────────────
# 🚀 Promote Model Version to Production
# ─────────────────────────────────────────────────────────────
def _promote_to_production(client, registry_name: str, version: str):
    """
    Promotes a specific version of a model to the 'Production' stage.
    Archives all existing production versions.
    """
    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Model '{registry_name}' version {version} promoted to 'Production'")

# ─────────────────────────────────────────────────────────────
# 🔐 Register and Promote Model or Pipeline
# ─────────────────────────────────────────────────────────────
def register_and_promote(
    registry_name: str,
    model_object=None,
    run_id: str = None,
    model_uri: str = None,
    is_pipeline: bool = False
):
    """
    Registers and promotes a model or preprocessor to MLflow Model Registry.

    Args:
        registry_name (str): Registry name to register the model under.
        model_object: The sklearn model or pipeline object (used if is_pipeline=True).
        run_id (str): MLflow run ID (used if is_pipeline=False).
        model_uri (str): Path to model artifact (e.g., 'runs:/<run_id>/model').
        is_pipeline (bool): Set True if logging a preprocessing pipeline.
    
    Returns:
        None
    """
    client = MlflowClient()
    _ensure_model_registered(client, registry_name)

    # ── Registering a preprocessing pipeline ──
    if is_pipeline:
        mlflow.set_experiment("Preprocessing Pipeline Registry")
        with mlflow.start_run(run_name=f"{registry_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                mlflow.sklearn.log_model(
                    sk_model=model_object,
                    artifact_path="preprocessor",
                    registered_model_name=registry_name
                )
                print(f"✅ Preprocessor registered as '{registry_name}'")
            except Exception as e:
                print(f"❌ Failed to log/register preprocessor: {e}")
                return

        # Promote the latest unpromoted version to Production
        versions = client.get_latest_versions(registry_name, stages=["None"])
        if not versions:
            print("⚠️ No unpromoted versions found.")
            return

        latest_version = versions[0].version
        _promote_to_production(client, registry_name, latest_version)

    # ── Registering a trained model ──
    else:
        if not run_id or not model_uri:
            raise ValueError("❌ 'run_id' and 'model_uri' must be provided for trained model registration.")

        try:
            mv = client.create_model_version(
                name=registry_name,
                source=model_uri,
                run_id=run_id
            )
        except MlflowException as e:
            if "not found" in str(e).lower():
                print(f"ℹ️ Model '{registry_name}' not found. Creating it...")
                client.create_registered_model(registry_name)
                mv = client.create_model_version(
                    name=registry_name,
                    source=model_uri,
                    run_id=run_id
                )
            else:
                raise

        _promote_to_production(client, registry_name, mv.version)
