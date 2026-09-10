"""
Turn seven detector scores into one risk score.

    risk = worst_signal * MAX_BLEND + total_signals * SUM_BLEND

Read it as: take the single worst signal, that is most of the score,
then add a small bonus if other detectors also fired.

Why not an average? A perfect ghost work would score 1.0/7 = 0.14 and
never surface. Most frauds trip exactly one detector; averaging lets
six silent detectors vote down the one that found something.

Why not a plain sum? Seven weak hints of 0.2 would sum to 1.4 and
outrank one certainty at 1.0.
"""

import pandas as pd
import config as C


def combine(scores):
    """scores: DataFrame, one column per detector, one row per work."""
    weighted = scores * pd.Series(C.WEIGHTS)
    risk = (weighted.max(axis=1) * C.MAX_BLEND
            + weighted.sum(axis=1).clip(0, 1) * C.SUM_BLEND)
    return risk.round(3), weighted
