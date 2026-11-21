"""Small statistics helpers shared by the notebooks and the dashboard."""

import numpy as np


def wilson_interval(defaults, loans, z=1.96):
    """A 95% confidence interval for a proportion (the Wilson interval)."""
    p = defaults / loans
    centre = (p + z**2 / (2 * loans)) / (1 + z**2 / loans)
    margin = z * np.sqrt(p * (1 - p) / loans + z**2 / (4 * loans**2)) / (1 + z**2 / loans)
    return centre - margin, centre + margin


def rate_table(frame, by):
    """Loans, defaults, default rate and 95% interval for each value of one or more columns."""
    table = frame.groupby(by, observed=True)["Default"].agg(loans="size", defaults="sum")
    table["default_rate"] = table["defaults"] / table["loans"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["defaults"], table["loans"])
    return table.reset_index()
