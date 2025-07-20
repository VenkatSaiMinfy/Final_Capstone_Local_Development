import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies domain-specific feature engineering
    to the lead scoring dataset.
    Adds:
        - Flags for zero values in behavioral columns
        - A composite engagement score
    """

    def __init__(self):
        pass

    def add_behavioral_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds binary flag columns indicating if key behavioral features are zero.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Modified dataframe with new flag columns.
        """
        for col in ['TotalVisits', 'Total Time Spent on Website']:
            if col in df.columns:
                df[f'{col}_is_zero'] = (df[col] == 0).astype(int)
        return df

    def add_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a new feature 'EngagementScore' as a product of two existing features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Modified dataframe with the combined feature.
        """
        if {'TotalVisits', 'Total Time Spent on Website'}.issubset(df.columns):
            df['EngagementScore'] = df['TotalVisits'] * df['Total Time Spent on Website']
        return df

    def fit(self, X: pd.DataFrame, y=None):
        """
        No fitting needed; this transformer is stateless.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies behavioral flag creation and combined feature addition.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe with new features.
        """
        X_copy = X.copy()
        X_copy = self.add_behavioral_flags(X_copy)
        X_copy = self.add_combined_features(X_copy)
        return X_copy
