from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import Optional, Union, Dict
import numpy as np


def compute_metrics(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    y_prob: Optional[Union[list, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for binary classification.

    Args:
        y_true (list or np.ndarray): True class labels.
        y_pred (list or np.ndarray): Predicted class labels.
        y_prob (list or np.ndarray, optional): Predicted probabilities for positive class.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score, and optionally roc_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0  # Handle cases where y_true has only one class

    return metrics
