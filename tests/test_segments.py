"""Checks that the dashboard treats a new borrower exactly like the historical loans.

Run from the project root:   python tests/test_segments.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import appdata, preprocessing as pp, stats
from src.config import DATA_DIR
from src.segments import SegmentModel, borrower_frame, category_options

MODEL = SegmentModel()


def test_assignment_matches_the_stored_labels():
    """Rebuild 500 real borrowers from their raw inputs and compare with what notebooks 06 and 09 stored."""
    loans = pd.read_csv(DATA_DIR / "loans_clean.csv")
    labels = np.load(DATA_DIR / "labels_kmeans.npy")
    flags = np.load(DATA_DIR / "anomaly_flags.npy")
    chosen = np.random.default_rng(1).choice(len(loans), 500, replace=False)

    result, _ = MODEL.assign(loans.iloc[chosen])
    # Saved matrices were stored as 32-bit numbers, so a borrower sitting exactly on a boundary could differ
    segment_mismatches = (result["Segment"].to_numpy() != labels[chosen]).sum()
    flag_mismatches = (result["IsAnomaly"].to_numpy() != flags[chosen]).sum()
    assert segment_mismatches <= 1, f"{segment_mismatches} segments differ from training"
    assert flag_mismatches <= 1, f"{flag_mismatches} anomaly flags differ from training"


def test_unseen_purpose_and_missing_employment_still_work():
    options = category_options(MODEL)
    borrower = borrower_frame(
        60000, 18, 12000, 700, None, "never_seen_before", options["HomeOwnership"][0]
    )
    result, distances = MODEL.assign(borrower)
    assert 0 <= int(result.loc[0, "Segment"]) < distances.shape[1]
    assert np.isfinite(distances).all()


def test_dti_above_100_is_treated_like_a_missing_value():
    options = category_options(MODEL)
    purpose, home = options["Purpose"][0], options["HomeOwnership"][0]
    impossible = borrower_frame(60000, 999, 12000, 700, 5, purpose, home)
    missing = borrower_frame(60000, np.nan, 12000, 700, 5, purpose, home)
    first, _ = MODEL.assign(impossible)
    second, _ = MODEL.assign(missing)
    assert first.loc[0, "Segment"] == second.loc[0, "Segment"]
    assert np.isclose(first.loc[0, "DistanceToCentre"], second.loc[0, "DistanceToCentre"])


def test_closeness_ratio_is_between_zero_and_one():
    loans = pd.read_csv(DATA_DIR / "loans_clean.csv").sample(200, random_state=2)
    result, _ = MODEL.assign(loans)
    assert result["ClosenessRatio"].between(0, 1).all()


def test_category_options_are_not_empty():
    options = category_options(MODEL)
    assert len(options["Purpose"]) > 1 and len(options["HomeOwnership"]) > 1


def test_wilson_interval_known_value():
    low, high = stats.wilson_interval(50, 100)
    assert abs(low - 0.4038) < 0.001 and abs(high - 0.5962) < 0.001


def test_rate_table():
    frame = pd.DataFrame({"Segment": [0, 0, 1, 1, 1], "Default": [1, 0, 0, 0, 1]})
    table = stats.rate_table(frame, "Segment").set_index("Segment")
    assert table.loc[0, "default_rate"] == 0.5
    assert abs(table.loc[1, "default_rate"] - 1 / 3) < 1e-9
    assert (table["ci_low"] < table["default_rate"]).all()


def test_filter_sample():
    sample = pd.DataFrame(
        {
            "Segment": [0, 1, 1],
            "IssueYear": [2014, 2015, 2016],
            "Purpose": ["a", "b", "b"],
            "HomeOwnership": ["x", "x", "y"],
            "State": ["CA", "TX", "TX"],
        }
    )
    assert len(appdata.filter_sample(sample, [0, 1], (2014, 2016), [], [], [])) == 3
    assert len(appdata.filter_sample(sample, [1], (2015, 2016), ["b"], [], [])) == 2
    assert len(appdata.filter_sample(sample, [0, 1], (2014, 2016), [], ["y"], ["TX"])) == 1


def test_preprocessing_rule_for_dti():
    frame = pd.DataFrame(
        [
            {
                "Income": 1,
                "DTI": 150,
                "LoanAmount": 1,
                "CreditScore": 700,
                "EmploymentLength": 1,
                "Purpose": "a",
                "HomeOwnership": "b",
            }
        ]
    )
    assert np.isnan(pp.prepare_features(frame).loc[0, "DTI"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"All {len(tests)} tests passed")
