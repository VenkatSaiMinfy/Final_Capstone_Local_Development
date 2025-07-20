from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CleanColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop irrelevant or redundant columns
    from the lead scoring dataset during preprocessing.
    """

    def __init__(self):
        self.drop_cols = [
            'Prospect ID', 'Lead Number', 'Magazine',
            'Receive More Updates About Our Courses',
            'Update me on Supply Chain Content',
            'Get updates on DM Content',
            'I agree to pay the amount through cheque',
            'Newspaper Article', 'X Education Forums',
            'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
            'Last Notable Activity', 'Page Views Per Visit'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops specified columns if they exist in the input DataFrame.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed DataFrame with specified columns dropped.
        """
        return X.drop(columns=[col for col in self.drop_cols if col in X.columns], errors='ignore')


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add domain-specific engineered features
    like flags for zero visits/time and engagement score.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features:
        - Binary flags for zero values in 'TotalVisits' and 'Total Time Spent on Website'
        - 'EngagementScore' as a product of the above two

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed DataFrame with new features.
        """
        df = X.copy()

        # Add binary flags if columns exist
        for col in ['TotalVisits', 'Total Time Spent on Website']:
            if col in df.columns:
                df[f'{col}_is_zero'] = (df[col] == 0).astype(int)

        # Add engagement score if both components are available
        if {'TotalVisits', 'Total Time Spent on Website'}.issubset(df.columns):
            df['EngagementScore'] = df['TotalVisits'] * df['Total Time Spent on Website']

        return df
