"""
Detector tests.

Each detector gets a clean peer group plus one work altered in exactly
the way that detector hunts. If a test fails you know which field broke
it, because only one field changed.
"""

from datetime import timedelta

import pandas as pd
import pytest

import config as C
import detectors as D
from conftest import work, frame, peers, SANCTION


# ═════════════════════════════════════════════ nothing fires on clean data

@pytest.mark.parametrize("name", [
    "ghost_work", "duplicate", "unit_cost",
    "cost_overrun", "stalled_work",
])
def test_clean_register_fires_nothing(clean_register, name):
    fn, _ = D.DETECTORS[name]
    scores, _ = fn(clean_register)
    assert scores.max() == 0, (
        f"{name} fired on a register with nothing wrong in it")


# ═════════════════════════════════════════════ ghost work

def test_ghost_work_catches_impossible_build():
    rows = peers(12) + [work(
        work_id="GHOST/1",
        work_start_date=SANCTION + timedelta(days=2),
        completion_date=SANCTION + timedelta(days=7),
        expenditure=1_670_000.0,          # 98% paid
        status="Completed")]
    scores, ev = D.d_ghost_work(frame(rows))
    assert scores.iloc[-1] == 1.0
    assert ev.iloc[-1]["days_sanction_to_completion"] == 7
    assert ev.iloc[-1]["paid_fraction"] > C.GHOST_MIN_PAID


def test_ghost_work_ignores_fast_but_unpaid():
    """Fast with no money released is a clerical slip, not a theft."""
    rows = peers(12) + [work(
        work_id="FAST/1",
        work_start_date=SANCTION + timedelta(days=2),
        completion_date=SANCTION + timedelta(days=7),
        expenditure=10_000.0,             # almost nothing paid
        status="Completed")]
    scores, _ = D.d_ghost_work(frame(rows))
    assert scores.iloc[-1] == 0.0


def test_ghost_work_ignores_incomplete_work():
    rows = peers(12) + [work(
        work_id="OPEN/1",
        work_start_date=SANCTION + timedelta(days=2),
        completion_date=SANCTION + timedelta(days=7),
        expenditure=1_670_000.0,
        status="In Progress")]
    scores, _ = D.d_ghost_work(frame(rows))
    assert scores.iloc[-1] == 0.0


# ═════════════════════════════════════════════ duplicate

def test_duplicate_catches_identical_description():
    desc = "Construction of pucca drain at Canal Head, Rajgir"
    rows = peers(10) + [
        work(work_id="A", work_description=desc,
             sanction_date=SANCTION, ward_or_village="Rajgir"),
        work(work_id="B", work_description=desc,
             sanction_date=SANCTION + timedelta(days=12),
             sanctioned_amount=1_712_000.0, revised_amount=1_712_000.0,
             ward_or_village="Rajgir"),
    ]
    scores, ev = D.d_duplicate(frame(rows))
    assert scores.iloc[-1] > 0.5
    assert ev.iloc[-1]["match_basis"] == "description"
    assert ev.iloc[-1]["text_similarity"] == 1.0


def test_duplicate_ignores_different_places():
    """Two genuine drains in one district. This is the case that used to
    produce 214 of 221 false positives."""
    # places deliberately absent from the peer fixture, so the only
    # possible match is A against B
    rows = peers(10) + [
        work(work_id="A",
             work_description="Construction of pucca drain at Canal Head, Rajgir",
             sanction_date=SANCTION, ward_or_village="Rajgir"),
        work(work_id="B",
             work_description="Construction of pucca drain at Forest Gate, Silao",
             sanction_date=SANCTION + timedelta(days=12),
             sanctioned_amount=1_712_000.0, revised_amount=1_712_000.0,
             ward_or_village="Silao"),
    ]
    scores, _ = D.d_duplicate(frame(rows))
    assert scores.iloc[-1] == 0.0
    assert scores.iloc[-2] == 0.0


def test_duplicate_ignores_pairs_outside_the_window():
    desc = "Construction of pucca drain at Canal Head, Rajgir"
    rows = peers(10) + [
        work(work_id="A", work_description=desc, sanction_date=SANCTION),
        work(work_id="B", work_description=desc,
             sanction_date=SANCTION + timedelta(days=C.DUP_WINDOW_DAYS + 30)),
    ]
    scores, _ = D.d_duplicate(frame(rows))
    assert scores.iloc[-1] == 0.0


def test_duplicate_density_discount_shrinks_confidence():
    """A match in a small group is evidence. The same match in a big
    group is arithmetic, and must score lower."""
    desc = "Construction of pucca drain at Canal Head, Rajgir"

    def pair_score(group_size):
        rows = peers(group_size) + [
            work(work_id="A", work_description=desc, sanction_date=SANCTION),
            work(work_id="B", work_description=desc,
                 sanction_date=SANCTION + timedelta(days=10),
                 sanctioned_amount=1_712_000.0, revised_amount=1_712_000.0),
        ]
        s, _ = D.d_duplicate(frame(rows))
        return s.iloc[-1]

    assert pair_score(4) > pair_score(60)


# ═════════════════════════════════════════════ unit cost

def test_unit_cost_catches_inflated_rate():
    rows = peers(14) + [work(
        work_id="EXPENSIVE/1",
        quantity=500.0,
        sanctioned_amount=1_700_000.0 * 4,
        revised_amount=1_700_000.0 * 4)]
    scores, ev = D.d_unit_cost(frame(rows))
    assert scores.iloc[-1] > 0.5
    assert ev.iloc[-1]["ratio_to_median"] > 3
    assert ev.iloc[-1]["basis"] == "rate_per_unit"


def test_unit_cost_needs_enough_peers():
    """Two works cannot establish a median worth judging against."""
    rows = peers(2) + [work(work_id="X", quantity=500.0,
                            sanctioned_amount=20_000_000.0)]
    scores, _ = D.d_unit_cost(frame(rows))
    assert scores.max() == 0


def test_unit_cost_falls_back_without_quantity(clean_register):
    """The real export has no quantity column. The detector must still run."""
    df = clean_register.drop(columns=["quantity"])
    odd = work(work_id="BIG/1", sanctioned_amount=30_000_000.0)
    del odd["quantity"]
    df = frame(df.to_dict("records") + [odd])
    scores, ev = D.d_unit_cost(df)
    assert scores.iloc[-1] > 0
    assert ev.iloc[-1]["basis"] == "sanctioned_amount"


# ═════════════════════════════════════════════ cost overrun

def test_cost_overrun_scales_with_size():
    small = frame(peers(8) + [work(work_id="S", revised_amount=1_700_000 * 1.4)])
    large = frame(peers(8) + [work(work_id="L", revised_amount=1_700_000 * 2.8)])
    s_small, _ = D.d_cost_overrun(small)
    s_large, _ = D.d_cost_overrun(large)
    assert s_large.iloc[-1] > s_small.iloc[-1]
    assert s_large.iloc[-1] == 1.0


def test_cost_overrun_ignores_small_revision():
    df = frame(peers(8) + [work(work_id="OK", revised_amount=1_700_000 * 1.1)])
    scores, _ = D.d_cost_overrun(df)
    assert scores.iloc[-1] == 0.0


# ═════════════════════════════════════════════ stalled work

def test_stalled_work_catches_old_unpaid_sanction():
    old = C.TODAY - timedelta(days=900)
    df = frame(peers(8) + [work(
        work_id="STALLED/1", sanction_date=old,
        work_start_date=None, completion_date=None,
        expenditure=0.0, status="In Progress")])
    scores, ev = D.d_stalled_work(df)
    assert scores.iloc[-1] > 0.5
    assert ev.iloc[-1]["amount_paid"] == 0.0
    assert ev.iloc[-1]["days_past_one_year_limit"] > 500


def test_stalled_work_ignores_recent_sanction():
    recent = C.TODAY - timedelta(days=120)
    df = frame(peers(8) + [work(
        work_id="NEW/1", sanction_date=recent,
        completion_date=None, expenditure=0.0, status="In Progress")])
    scores, _ = D.d_stalled_work(df)
    assert scores.iloc[-1] == 0.0


# ═════════════════════════════════════════════ round numbers

def test_round_number_scales_with_roundness():
    df = frame([
        work(work_id="A", sanctioned_amount=1_713_000.0),   # not round
        work(work_id="B", sanctioned_amount=1_700_000.0),   # 1 lakh
        work(work_id="C", sanctioned_amount=1_500_000.0),   # 5 lakh
        work(work_id="D", sanctioned_amount=2_000_000.0),   # 10 lakh
    ])
    s, _ = D.d_round_number(df)
    assert s.iloc[0] < s.iloc[1] < s.iloc[2] < s.iloc[3]


def test_round_number_alone_never_flags_a_work():
    """On real MPLADS data a quarter of amounts are exact multiples of
    Rs 5 lakh. If this detector could fire alone it would bury an auditor
    in honest works."""
    import scoring as S
    df = frame([work(work_id="D", sanctioned_amount=5_000_000.0)])
    raw, _ = D.d_round_number(df)
    scores = pd.DataFrame({k: [0.0] for k in C.WEIGHTS})
    scores["round_number"] = raw.values
    risk, _ = S.combine(scores)
    assert risk.iloc[0] < C.RISK_THRESHOLD


# ═════════════════════════════════════════════ agency capture

def test_agency_capture_fires_on_a_dominated_district():
    rows = peers(6, implementing_agency="Panchayat Samiti Cell")
    rows += peers(2, implementing_agency="PWD Division")
    for i, r in enumerate(rows):
        r["work_id"] = f"W{i}"
    scores, ev = D.d_agency_capture(frame(rows))
    assert scores.max() > 0
    hit = ev[scores > 0].iloc[0]
    assert hit["dominant_agency"] == "Panchayat Samiti Cell"
    assert hit["hhi"] > C.HHI_FLOOR


def test_agency_capture_quiet_on_a_balanced_district():
    agencies = ["PWD Division", "Zilla Parishad Works",
                "Jal Nigam Unit", "Block Development Office"]
    rows = []
    for i in range(16):
        rows.append(work(work_id=f"W{i}",
                         implementing_agency=agencies[i % 4]))
    scores, _ = D.d_agency_capture(frame(rows))
    assert scores.max() == 0


@pytest.mark.parametrize("det", sorted(C.CORROBORATING))
def test_corroborating_detector_cannot_surface_a_work_alone(det):
    """The design rule, enforced in scoring.combine rather than trusted.
    A corroborating signal adds weight to someone else's finding; it must
    never flag a work by itself."""
    import scoring as S
    cap = S.corroborating_cap(det)
    weighted = cap * C.WEIGHTS[det]
    composite = weighted * C.MAX_BLEND + weighted * C.SUM_BLEND
    assert composite < C.RISK_THRESHOLD, (
        f"{det} alone reaches {composite:.3f}, at or above the "
        f"{C.RISK_THRESHOLD} threshold")


# ═════════════════════════════════════════════ self-skipping

@pytest.mark.parametrize("name,missing", [
    ("ghost_work", "completion_date"),
    ("cost_overrun", "revised_amount"),
    ("stalled_work", "expenditure"),
])
def test_detector_skips_loudly_when_a_column_is_missing(name, missing,
                                                        clean_register):
    """A detector that cannot run must say so, not return silent zeros."""
    df = clean_register.drop(columns=[missing])
    fn, _ = D.DETECTORS[name]
    scores, ev = fn(df)
    assert scores.max() == 0
    assert "skipped" in ev.iloc[0], (
        f"{name} returned zeros without explaining why")