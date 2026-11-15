"""Preprocessing for borrower segmentation.

One function, build_preprocessor(), returns the whole pipeline. The notebooks and the
dashboard both use it, so a new borrower is treated exactly like the historical loans.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DTI_MAX = 100                        # larger values (such as the code 999) are not real ratios

LOG_COLUMNS = ["Income", "LoanAmount"]                       # right-skewed: log first
PLAIN_NUMERIC_COLUMNS = ["DTI", "CreditScore", "EmploymentLength"]
CATEGORICAL_COLUMNS = ["Purpose", "HomeOwnership"]
FEATURE_COLUMNS = LOG_COLUMNS + PLAIN_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

CATEGORICAL_WEIGHT = 0.5             # one-hot columns count half as much as numeric ones
RARE_CATEGORY_SHARE = 0.02           # categories under 2% of loans are grouped together


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Cap extreme values at chosen percentiles learned from the data."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = pd.DataFrame(X)
        self.low_ = values.quantile(self.lower).to_numpy()
        self.high_ = values.quantile(self.upper).to_numpy()
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def prepare_features(frame):
    """Select the clustering columns and turn impossible DTI values into missing values."""
    features = frame[FEATURE_COLUMNS].copy()
    features.loc[features["DTI"] > DTI_MAX, "DTI"] = np.nan
    return features


def build_preprocessor(categorical_weight=CATEGORICAL_WEIGHT):
    """Log/clip/fill/scale the numbers and one-hot encode the categories."""
    log_numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("clip", PercentileClipper()),
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    plain_numeric = Pipeline([
        ("clip", PercentileClipper()),
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=RARE_CATEGORY_SHARE, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("log_numeric", log_numeric, LOG_COLUMNS),
         ("numeric", plain_numeric, PLAIN_NUMERIC_COLUMNS),
         ("categorical", categorical, CATEGORICAL_COLUMNS)],
        transformer_weights={"log_numeric": 1.0, "numeric": 1.0, "categorical": categorical_weight},
        verbose_feature_names_out=False,
    )