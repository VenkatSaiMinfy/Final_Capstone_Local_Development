from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Tuple, Union, List


def apply_feature_selection(
    X: Union[pd.DataFrame, pd.Series],
    y: Union[pd.Series, list],
    top_n: int = 50
) -> Tuple[Union[pd.DataFrame, pd.Series], List[int]]:
    """
    Applies Recursive Feature Elimination (RFE) using RandomForestClassifier
    to select the top `top_n` features from the dataset.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or list): Target vector.
        top_n (int): Number of top features to select.

    Returns:
        Tuple:
            - X_selected (pd.DataFrame or np.ndarray): Dataset with selected features.
            - selected_indices (List[int]): List of selected feature indices.
    """
    if top_n > X.shape[1]:
        raise ValueError(f"top_n={top_n} cannot be greater than number of features ({X.shape[1]})")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(model, n_features_to_select=top_n)
    selector.fit(X, y)

    selected_indices = list(selector.get_support(indices=True))

    if isinstance(X, pd.DataFrame):
        X_selected = X.iloc[:, selected_indices]
    else:
        X_selected = selector.transform(X)

    return X_selected, selected_indices
