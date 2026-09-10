"""
The shape of the data.

Definitions only. Nothing here calculates anything.
This file is the contract: every data source must produce these
columns, and every detector may assume they exist.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


# ─────────────────────────────────────────────────────────────
# THE INPUT CONTRACT
# ─────────────────────────────────────────────────────────────

CORE_COLUMNS = [
    "work_id", "state", "district", "constituency",
    "work_name", "work_description",
    "ward_or_village", "implementing_agency",
    "recommendation_date", "sanctioned_amount", "status",
]

# Columns some sources have and some don't.
# The real eSAKSHI scrape is missing most of these.
OPTIONAL_COLUMNS = [
    "pfms_code", "unit", "quantity", "location_type",
    "sanction_date", "work_start_date", "completion_date",
    "recommended_amount", "revised_amount", "expenditure",
    "vendor_name", "sc_st_component", "ida_approval",
]

DATE_COLUMNS = [
    "recommendation_date", "sanction_date",
    "work_start_date", "completion_date",
]

MONEY_COLUMNS = [
    "recommended_amount", "sanctioned_amount",
    "revised_amount", "expenditure",
]

GROUND_TRUTH_COLUMN = "fraud_label"   # synthetic data only


def validate(df):
    """Stop early if the core columns are missing.

    A missing column caught here gives a clear error.
    The same column missing silently gives you a detector
    quietly returning zeros three files away.
    """
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}"
        )
    return True


def available(df, columns):
    """True if every column in `columns` exists AND has some data.

    This is how a detector decides whether it can run at all.
    """
    for c in columns:
        if c not in df.columns:
            return False
        if df[c].notna().sum() == 0:
            return False
    return True


# ─────────────────────────────────────────────────────────────
# THE OUTPUT SHAPES
# ─────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """One detector's opinion about one work, with its working shown.

    The score ranks the work. The evidence is what an auditor reads.
    Both matter, but only one of them is a product.
    """
    detector: str             # "unit_cost"
    score: float              # 0.0 to 1.0
    fired: bool
    headline: str             # one plain sentence with real numbers
    evidence: dict            # raw numbers ONLY, never formatted strings

    def __repr__(self):
        return f"<Signal {self.detector} {self.score:.2f}>"


@dataclass
class FlaggedWork:
    """Everything the dashboard and the audit-note layer need."""

    work_id: str
    state: str
    district: str
    work_name: str
    work_description: str
    implementing_agency: str
    ward_or_village: str
    sanctioned_amount: float

    risk_score: float
    signals: list = field(default_factory=list)

    @property
    def primary_signal(self):
        """The detector that drove the score."""
        return max(self.signals, key=lambda s: s.score)

    @property
    def fired_signals(self):
        """Only detectors that actually fired, worst first."""
        return sorted([s for s in self.signals if s.fired],
                      key=lambda s: s.score, reverse=True)

    def to_dict(self):
        return asdict(self)
