import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features: List[int]):
        """
        A custom transformer to select features by index.

        Args:
            selected_features (List[int]): List of column indices to keep.
        """
        self.selected_features = selected_features

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Selects features from the input X based on the indices provided.

        Returns:
            np.ndarray or pd.DataFrame: Subset with only selected columns.
        """
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features]
        else:
            X = np.asarray(X)
            return X[:, self.selected_features]
