import pandas as pd
from mlflow.tracking import MlflowClient


def get_run_metrics(run_id: str, client: MlflowClient) -> dict:
    """
    Fetch all metrics and parameters for a specific MLflow run.

    Args:
        run_id (str): The unique run ID.
        client (MlflowClient): Active MLflow client instance.

    Returns:
        dict: Combined dictionary of metrics and parameters.
    """
    run_data = client.get_run(run_id).data
    return {**run_data.metrics, **run_data.params}


def compare_models(experiment_name: str = "Lead Scoring Models") -> pd.DataFrame:
    """
    Compare ML models from a specific MLflow experiment by aggregating key metrics.

    Args:
        experiment_name (str): Name of the MLflow experiment to search in.

    Returns:
        pd.DataFrame: Sorted dataframe of all runs by descending F1-score.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"❌ Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000
    )

    records = []
    for run in runs:
        run_id = run.info.run_id
        run_data = {
            "run_id": run_id,
            "model_name": run.data.tags.get("mlflow.runName", "Unnamed Run")
        }
        metrics_params = get_run_metrics(run_id, client)
        run_data.update(metrics_params)
        records.append(run_data)

    df = pd.DataFrame(records)

    # Convert numeric columns for sorting (safe even if some are missing)
    for col in ['accuracy', 'f1_score', 'roc_auc', 'cv_mean_test_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_sorted = df.sort_values(by="f1_score", ascending=False)
    return df_sorted.reset_index(drop=True)
