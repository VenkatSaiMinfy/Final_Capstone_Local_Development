import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Add project root to path for imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import custom feature engineering transformer
from src.ml.pipeline.feature_engineering import FeatureEngineeringTransformer

# ─────────────────────────────────────────────────────────────
# 🧹 Step 1: Column Cleaning Function
# ─────────────────────────────────────────────────────────────

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unnecessary, constant, or identifier columns that do not add predictive value.
    
    Parameters:
        df (pd.DataFrame): Raw input DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    drop_cols = [
        'Prospect ID', 'Lead Number', 'Magazine',
        'Receive More Updates About Our Courses',
        'Update me on Supply Chain Content',
        'Get updates on DM Content',
        'I agree to pay the amount through cheque',
        'Newspaper Article', 'X Education Forums',
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
        'Last Notable Activity', 'Page Views Per Visit'
    ]

    # Drop columns only if they exist in the DataFrame
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df


# ─────────────────────────────────────────────────────────────
# ⚙️ Step 2: Column-wise Preprocessing Pipelines
# ─────────────────────────────────────────────────────────────

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Constructs column-wise preprocessing logic using sklearn pipelines.
    
    Parameters:
        numeric_features (list[str]): List of numeric column names
        categorical_features (list[str]): List of categorical column names

    Returns:
        ColumnTransformer: Combined preprocessing pipeline
    """

    # Numeric pipeline: Impute missing values with median, then scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: Impute with most frequent, then apply OneHotEncoding
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine both into a single preprocessor
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# ─────────────────────────────────────────────────────────────
# 🔄 Step 3: Full Preprocessing Pipeline (Feature Engg + Cleaning)
# ─────────────────────────────────────────────────────────────

def get_full_pipeline(numeric_features, categorical_features):
    """
    Constructs the complete preprocessing pipeline, including:
    - Feature engineering
    - Column-wise preprocessing (imputation + scaling/encoding)

    Parameters:
        numeric_features (list[str]): List of numeric column names
        categorical_features (list[str]): List of categorical column names

    Returns:
        Pipeline: A complete sklearn pipeline
    """
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)

    full_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),  # Adds behavioral flags and new features
        ("preprocessing", preprocessor)                            # Handles imputation, scaling, encoding
    ])

    return full_pipeline
