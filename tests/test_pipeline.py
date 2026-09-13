"""
Tests for the layers above the detectors: how scores are blended, whether
the guideline rules fire, whether the generator is reproducible, and
whether end-to-end accuracy has regressed.
"""

import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

import compliance as K
import config as C
import engine as E
import models as M
import scoring as S
from conftest import work, frame, peers, SANCTION


def blank_scores(n=1, **vals):
    d = {k: [0.0] * n for k in C.WEIGHTS}
    for k, v in vals.items():
        d[k] = [v] * n if not isinstance(v, list) else v
    return pd.DataFrame(d)


# ═════════════════════════════════════════════ blending

def test_one_certain_signal_surfaces_a_work():
    """A confirmed ghost work trips one detector and six stay silent.
    Averaging would score it 1/7 = 0.14 and bury it."""
    risk, _ = S.combine(blank_scores(ghost_work=1.0))
    assert risk.iloc[0] > C.RISK_THRESHOLD
    assert risk.iloc[0] == pytest.approx(1.0)


def test_corroboration_raises_the_score():
    """Same primary finding, more supporting evidence, higher rank."""
    alone, _ = S.combine(blank_scores(duplicate=0.9))
    backed, _ = S.combine(blank_scores(duplicate=0.9, unit_cost=0.6))
    assert backed.iloc[0] > alone.iloc[0]


def test_many_weak_signals_do_not_beat_one_certainty():
    """Seven maybes should not outrank one proof."""
    certain, _ = S.combine(blank_scores(ghost_work=1.0))
    vague, _ = S.combine(blank_scores(
        duplicate=0.25, unit_cost=0.25, cost_overrun=0.25,
        stalled_work=0.25, round_number=0.25))
    assert certain.iloc[0] > vague.iloc[0]


def test_score_is_bounded():
    risk, _ = S.combine(blank_scores(**{k: 1.0 for k in C.WEIGHTS}))
    assert 0.0 <= risk.iloc[0] <= 1.0


def test_nothing_firing_scores_zero():
    risk, _ = S.combine(blank_scores())
    assert risk.iloc[0] == 0.0


def test_weights_are_ordered_by_strength_of_evidence():
    """Ghost work proves money left for nothing. Agency concentration
    proves nothing at all. The order encodes that."""
    w = C.WEIGHTS
    assert w["ghost_work"] > w["duplicate"] > w["unit_cost"]
    assert w["unit_cost"] > w["stalled_work"]
    assert w["round_number"] < w["ghost_work"]


# ═════════════════════════════════════════════ compliance rules

def test_sanction_sla_fires_past_45_days():
    as_of = SANCTION + timedelta(days=200)
    df = frame([work(work_id="LATE", recommendation_date=SANCTION,
                     sanction_date=None, status="Unsanctioned")])
    b, notes, skipped, used = K.check(df, as_of=as_of)
    assert b["sanction_sla"].iloc[0]
    assert "\u00a73.2.4" in notes.iloc[0][0]["clause"]


def test_sanction_sla_quiet_inside_45_days():
    as_of = SANCTION + timedelta(days=20)
    df = frame([work(work_id="OK", recommendation_date=SANCTION,
                     sanction_date=None, status="Unsanctioned")])
    b, _, _, _ = K.check(df, as_of=as_of)
    assert not b["sanction_sla"].iloc[0]


def test_completion_limit_fires_past_one_year():
    as_of = SANCTION + timedelta(days=500)
    df = frame([work(work_id="OVER", completion_date=None,
                     status="In Progress")])
    b, notes, _, _ = K.check(df, as_of=as_of)
    assert b["completion_limit"].iloc[0]
    assert "\u00a73.2.13" in notes.iloc[0][0]["clause"]


def test_completion_limit_quiet_for_finished_work():
    as_of = SANCTION + timedelta(days=500)
    df = frame([work(work_id="DONE", status="Completed")])
    b, _, _, _ = K.check(df, as_of=as_of)
    assert not b["completion_limit"].iloc[0]


def test_minimum_sanction_is_not_called_a_violation():
    """The Guideline says works shall NORMALLY be Rs 2.5 lakh or more, and
    permits less for reasons recorded. Calling that a violation would
    overclaim on 19,735 real works."""
    assert K.RULES["minimum_sanction"]["severity"] == "NEEDS JUSTIFICATION"
    assert K.RULES["sanction_sla"]["severity"] == "VIOLATION"


def test_minimum_sanction_fires_below_the_floor():
    df = frame([work(work_id="SMALL", sanctioned_amount=100_000.0)])
    b, notes, _, _ = K.check(df)
    assert b["minimum_sanction"].iloc[0]
    assert "justification" in notes.iloc[0][0]["text"].lower()


def test_compliance_measures_against_the_snapshot_not_today():
    """A 2024 export judged against today would report every pending work
    as years overdue — a fact about the download date, not the ministry."""
    df = frame(peers(4))
    _, _, _, as_of = K.check(df)
    assert as_of == df.completion_date.max()


def test_compliance_rule_skips_when_a_column_is_missing():
    df = frame(peers(4)).drop(columns=["recommendation_date"])
    _, _, skipped, _ = K.check(df)
    assert any(name == "sanction_sla" for name, _ in skipped)


# ═════════════════════════════════════════════ the generator

def test_generator_is_reproducible():
    """Seed-fixed, so anyone can regenerate the identical file and check
    the accuracy claim."""
    import generator as G
    a = G.build(n=400)
    b = G.build(n=400)
    assert len(a) == len(b)
    assert a.fraud_label.value_counts().to_dict() == \
        b.fraud_label.value_counts().to_dict()


def test_every_planted_fraud_is_labelled(synthetic):
    df, _, _, _ = synthetic
    planted = df[df.fraud_label != "CLEAN"]
    assert len(planted) > 500
    assert planted.fraud_label.str.strip().ne("").all()


def test_ghost_works_are_internally_coherent(synthetic):
    """An early generator logged completion BEFORE the start date on 44 of
    45 ghost works. The detector was catching a data bug, not fraud."""
    df, _, _, _ = synthetic
    g = df[df.fraud_label.str.contains("GHOST_WORK", na=False)]
    dur = (g.completion_date - g.work_start_date).dt.days
    assert (dur >= 0).all(), "a work completed before it started"
    assert (dur < C.GHOST_MAX_DAYS).all()


def test_descriptions_are_sometimes_missing(synthetic):
    """The eSAKSHI field is optional. If every synthetic work had one, the
    duplicate detector would score well here and fail in production."""
    df, _, _, _ = synthetic
    present = df.work_description.fillna("").ne("").mean()
    assert 0.5 < present < 0.9


def test_engine_never_reads_the_label():
    """The one guarantee the accuracy number rests on.

    Locates each module through the import system rather than by walking
    up from this file, so the test works wherever pytest is run from.
    """
    import detectors, engine, ml, scoring

    for mod in (detectors, scoring, engine, ml):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert M.GROUND_TRUTH_COLUMN not in src, (
            f"{Path(mod.__file__).name} references "
            f"{M.GROUND_TRUTH_COLUMN} \u2014 the engine must never see "
            f"the answer key")


# ═════════════════════════════════════════════ contract

def test_validate_rejects_a_missing_core_column(clean_register):
    with pytest.raises(ValueError, match="missing required columns"):
        M.validate(clean_register.drop(columns=["district"]))


def test_validate_accepts_a_complete_frame(clean_register):
    assert M.validate(clean_register) is True


def test_available_is_false_for_an_all_null_column(clean_register):
    df = clean_register.copy()
    df["expenditure"] = None
    assert not M.available(df, ["expenditure"])


# ═════════════════════════════════════════════ end to end

def test_engine_runs_and_shapes_are_consistent(synthetic):
    df, scores, evidence, _ = synthetic
    assert len(scores) == len(df)
    assert set(scores.columns) == set(C.WEIGHTS)
    assert df.risk_score.between(0, 1).all()
    assert df.flagged.sum() > 0


def test_every_flagged_work_can_produce_an_audit_note(synthetic):
    """If a note ever raises, the dashboard breaks in front of a user."""
    import explain as X
    df, scores, evidence, _ = synthetic
    for idx in df[df.flagged].head(200).index:
        note = X.build_note(df.loc[idx], E.signals_for(df, scores, evidence, idx))
        assert note["headline"]
        assert note["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_every_fired_signal_states_real_numbers(synthetic):
    """Evidence, not just a score. A headline with no figure in it cannot
    be acted on."""
    df, scores, evidence, _ = synthetic
    for idx in df[df.flagged].head(120).index:
        for s in E.signals_for(df, scores, evidence, idx):
            if not s.fired:
                continue
            assert s.evidence, f"{s.detector} fired with no evidence"
            assert any(ch.isdigit() for ch in s.headline), (
                f"{s.detector} headline states no number: {s.headline}")


# ═════════════════════════════════════════════ regression floor

# The numbers the README claims. If a change drops below these, the claim
# is no longer true and this test should fail before the README does.
MIN_RECALL = 0.90
MIN_PRECISION = 0.95


def test_accuracy_has_not_regressed(synthetic):
    df, _, _, _ = synthetic
    work_level = df[df.fraud_label != "AGENCY_CAPTURE"]
    truth = work_level.fraud_label != "CLEAN"
    flag = work_level.flagged

    recall = (flag & truth).sum() / max(int(truth.sum()), 1)
    precision = (flag & truth).sum() / max(int(flag.sum()), 1)

    assert recall >= MIN_RECALL, (
        f"recall fell to {recall:.1%}, below the {MIN_RECALL:.0%} floor "
        f"the README claims")
    assert precision >= MIN_PRECISION, (
        f"precision fell to {precision:.1%}, below the {MIN_PRECISION:.0%} "
        f"floor the README claims")


def test_false_alarm_count_stays_small(synthetic):
    df, _, _, _ = synthetic
    work_level = df[df.fraud_label != "AGENCY_CAPTURE"]
    fp = int((work_level.flagged & (work_level.fraud_label == "CLEAN")).sum())
    assert fp < 40, f"{fp} false alarms — the list is getting noisy"