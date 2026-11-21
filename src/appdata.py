"""Reading the small data files the dashboard uses, and filtering the sample."""

import pandas as pd

from src.config import APP_DATA_DIR


def read_table(name, **options):
    """Read a file from the app_data folder, or return None if it does not exist yet."""
    path = APP_DATA_DIR / name
    return pd.read_csv(path, **options) if path.exists() else None


def filter_sample(sample, segments, years, purposes, home_types, states):
    """Keep loans matching the sidebar choices. An empty list means 'do not filter on this'."""
    mask = sample["Segment"].isin(segments) & sample["IssueYear"].between(years[0], years[1])
    if purposes:
        mask &= sample["Purpose"].isin(purposes)
    if home_types:
        mask &= sample["HomeOwnership"].isin(home_types)
    if states:
        mask &= sample["State"].isin(states)
    return sample[mask]
