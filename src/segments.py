"""Place a new borrower into one of the saved segments and check whether the profile is unusual."""

import json

import joblib
import numpy as np
import pandas as pd

from src import preprocessing as pp
from src.config import MODEL_DIR


def borrower_frame(
    income, dti, loan_amount, credit_score, employment_length, purpose, home_ownership
):
    """A one-row table in the format the saved pipeline expects (employment length None = unknown)."""
    return pd.DataFrame(
        [
            {
                "Income": income,
                "DTI": dti,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "EmploymentLength": np.nan if employment_length is None else employment_length,
                "Purpose": purpose,
                "HomeOwnership": home_ownership,
            }
        ]
    )


class SegmentModel:
    """Loads the saved pipeline, PCA, K-Means and Isolation Forest from the models folder."""

    def __init__(self, model_dir=MODEL_DIR):
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.pca = joblib.load(model_dir / "pca.joblib")
        self.kmeans = joblib.load(model_dir / "kmeans.joblib")
        self.isolation = joblib.load(model_dir / "isolation_forest.joblib")
        self.n_components = self._read_json(model_dir / "feature_info.json")["n_components"]
        self.feature_space = self._read_json(model_dir / "clustering_info.json")["feature_space"]
        self.threshold = self._read_json(model_dir / "anomaly_info.json")["score_threshold"]

    @staticmethod
    def _read_json(path):
        with open(path) as file:
            return json.load(file)

    def transform(self, frame):
        """Apply exactly the preprocessing used on the historical loans."""
        scaled = self.preprocessor.transform(pp.prepare_features(frame))
        reduced = self.pca.transform(scaled)[:, : self.n_components]
        return scaled, reduced

    def assign(self, frame):
        """Return (a table with one row per borrower, the distance to every segment centre)."""
        scaled, reduced = self.transform(frame)
        space = reduced if self.feature_space == "PCA" else scaled
        distances = self.kmeans.transform(space)
        order = np.argsort(distances, axis=1)
        rows = np.arange(len(frame))
        nearest, second = order[:, 0], order[:, 1]

        result = pd.DataFrame(
            {
                "Segment": nearest,
                "DistanceToCentre": distances[rows, nearest],
                "DistanceToNextCentre": distances[rows, second],
            }
        )
        result["ClosenessRatio"] = result["DistanceToCentre"] / result["DistanceToNextCentre"]
        scores = self.isolation.score_samples(reduced)
        result["AnomalyScore"] = scores
        result["IsAnomaly"] = scores <= self.threshold
        return result, distances


def category_options(model):
    """The loan purposes and home-ownership types the saved pipeline knows about."""
    encoder = model.preprocessor.named_transformers_["categorical"].named_steps["encode"]
    return {
        "Purpose": [str(v) for v in encoder.categories_[0]],
        "HomeOwnership": [str(v) for v in encoder.categories_[1]],
    }
