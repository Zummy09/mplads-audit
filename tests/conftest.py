"""
Shared fixtures.

`work()` builds one deliberately boring, obviously-fine work. Each test
then changes only the field it is testing. That way a failure points at
one thing, instead of leaving you to work out which of 27 columns
mattered.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C          # noqa: E402
import engine as E          # noqa: E402
import models as M          # noqa: E402

SANCTION = pd.Timestamp("2024-06-01")


def work(**overrides):
    """One clean, unremarkable work. Override only what a test cares about."""
    base = {
        "work_id": "TEST/0001",
        "state": "Bihar",
        "district": "Nalanda",
        "constituency": "NALANDA",
        "mp_name": "Test MP",
        "pfms_code": "1.5.7",
        "work_category": "Drinking water and sanitation",
        "work_name": "Providing drains and gutters for public drainage",
        "annexure_ref": "5.7",
        "work_description": "Construction of pucca drain at Bus Stand, Rajgir",
        "unit": "metre",
        "quantity": 500.0,
        "location_type": "Rural",
        "ward_or_village": "Rajgir",
        "implementing_agency": "PWD Division",
        "vendor_name": "Test Works",
        "recommendation_date": SANCTION - timedelta(days=30),
        "sanction_date": SANCTION,
        "work_start_date": SANCTION + timedelta(days=20),
        "completion_date": SANCTION + timedelta(days=180),
        "recommended_amount": 1_700_000.0,
        "sanctioned_amount": 1_700_000.0,
        "revised_amount": 1_700_000.0,
        "expenditure": 1_650_000.0,
        "status": "Completed",
        "sc_st_component": False,
        "fraud_label": "CLEAN",
    }
    base.update(overrides)
    return base


def frame(rows):
    """Build a DataFrame with dates already parsed, as the engine expects."""
    df = pd.DataFrame(rows)
    for c in M.DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# Real works name real, different places. An early version of this fixture
# used "Point 1", "Point 2"... — which tokenises to the same words, so the
# duplicate detector flagged the whole peer group. It was right to.
PLACES = ["Bus Stand", "Panchayat Bhavan", "Primary School", "Main Chowk",
          "Health Centre", "Market Road", "Railway Crossing", "Temple Road",
          "Post Office", "Anganwadi Centre", "Police Chowki", "Water Tank",
          "High School", "Community Hall", "Ration Shop", "Bridge Point",
          "Bank Branch", "Fair Ground", "Cattle Shed", "Grain Store"]


def peers(n=12, **overrides):
    """A believable peer group, so median-based detectors have something
    to compare against. Amounts, dates and places all differ, as real
    ones do."""
    out = []
    for i in range(n):
        place = PLACES[i % len(PLACES)]
        out.append(work(
            work_id=f"PEER/{i:04d}",
            quantity=480.0 + i * 5,
            sanctioned_amount=1_650_000.0 + i * 9_000,
            revised_amount=1_650_000.0 + i * 9_000,
            expenditure=1_600_000.0 + i * 9_000,
            ward_or_village=f"Village {i}",
            work_description=f"Construction of pucca drain at {place}, Rajgir",
            sanction_date=SANCTION + timedelta(days=i * 40),
            work_start_date=SANCTION + timedelta(days=i * 40 + 20),
            completion_date=SANCTION + timedelta(days=i * 40 + 180),
            **overrides))
    return out


@pytest.fixture
def clean_register():
    """A register where nothing should fire."""
    return frame(peers(14))


@pytest.fixture(scope="session")
def synthetic():
    """The generated register, scored. Session-scoped: it is slow."""
    raw = pd.read_csv(C.SYNTHETIC_CSV)
    df, scores, evidence, skipped = E.run(raw)
    return df, scores, evidence, skipped