"""
How well is it working?

Only runs on synthetic data, because only synthetic data has an
answer key. Real MPLADS data has no fraud_label and never will.
"""

import pandas as pd

import config as C
import models as M
import engine as E


def report(df, scores, skipped):
    # agency_capture describes a DISTRICT, not a work. Scoring it as a
    # work-level detector would be measuring the wrong thing, so those
    # works are reported separately below.
    is_capture_only = df[M.GROUND_TRUTH_COLUMN] == "AGENCY_CAPTURE"
    work_level = df[~is_capture_only]
    wl_scores = scores.loc[work_level.index]

    truth = work_level[M.GROUND_TRUTH_COLUMN] != "CLEAN"
    flag = work_level.flagged

    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())

    recall = tp / max(int(truth.sum()), 1)
    precision = tp / max(int(flag.sum()), 1)

    print("=" * 62)
    print(f"WORK-LEVEL DETECTION  @ risk > {C.RISK_THRESHOLD}")
    print("=" * 62)
    print(f"  works                {len(work_level):>6,}")
    print(f"  planted frauds       {int(truth.sum()):>6,}")
    print(f"  surfaced for review  {int(flag.sum()):>6,}")
    print(f"  caught (true pos)    {tp:>6,}")
    print(f"  false alarms         {fp:>6,}")
    print(f"  missed               {fn:>6,}")
    print(f"\n  RECALL     {recall:6.1%}")
    print(f"  PRECISION  {precision:6.1%}")

    print("\n" + "=" * 62)
    print("DISTRICT-LEVEL DETECTION  (agency concentration)")
    print("=" * 62)
    cap = scores.agency_capture > 0
    found = sorted(df[cap].district.unique())
    planted = sorted(df[is_capture_only].district.unique())
    print(f"  districts flagged for concentration: {found}")
    print(f"  district actually captured:          {planted}")
    print(f"  works in flagged districts: {int(cap.sum()):,}")
    print("  (reported as a district risk view, not as individual flags)")

    if skipped:
        print("\n  detectors skipped (columns unavailable):")
        for n, why in skipped:
            print(f"    - {n}: {why}")

    print("\n" + "=" * 62)
    print("PER DETECTOR")
    print("=" * 62)
    label = {
        "ghost_work": "GHOST_WORK", "duplicate": "DUPLICATE_WORK",
        "unit_cost": "UNIT_COST_OUTLIER", "cost_overrun": "COST_OVERRUN",
        "stalled_work": "STALLED_WORK", "round_number": "ROUND_NUMBER",
        "agency_capture": "AGENCY_CAPTURE",
    }
    for det, tag in label.items():
        actual = df[M.GROUND_TRUTH_COLUMN].str.contains(tag, na=False)
        fired = scores[det] > C.FIRE_THRESHOLD
        caught = fired & actual
        r = caught.sum() / max(actual.sum(), 1)
        p = caught.sum() / max(fired.sum(), 1)
        print(f"  {tag:<20} planted {actual.sum():>4}  fired {fired.sum():>5}"
              f"   recall {r:6.1%}   precision {p:6.1%}")

    print("\n" + "=" * 62)
    print("FALSE ALARMS BY DRIVING DETECTOR")
    print("=" * 62)
    fpx = work_level[flag & ~truth]
    if len(fpx):
        for k, v in fpx.primary_detector.value_counts().items():
            print(f"  {k:<18} {v:>4}   {v/len(fpx):>5.1%}")
    else:
        print("  none")

    print("\n" + "=" * 62)
    print("WHAT THE ML LAYER ADDS")
    print("=" * 62)
    if "isolation_forest" in scores.columns:
        w = scores * pd.Series(C.WEIGHTS)
        rules_only = scores.drop(columns=["isolation_forest"])
        wr = rules_only * pd.Series(
            {k: v for k, v in C.WEIGHTS.items() if k != "isolation_forest"})
        risk_r = (wr.max(axis=1) * C.MAX_BLEND
                  + wr.sum(axis=1).clip(0, 1) * C.SUM_BLEND)
        fr = (risk_r > C.RISK_THRESHOLD)[work_level.index]
        tp_r = int((fr & truth).sum())
        print(f"  rules only   recall {tp_r/max(int(truth.sum()),1):6.1%}   "
              f"precision {tp_r/max(int(fr.sum()),1):6.1%}   "
              f"surfaced {int(fr.sum()):>4}")
        print(f"  with ML      recall {recall:6.1%}   "
              f"precision {precision:6.1%}   surfaced {int(flag.sum()):>4}")
        gained = work_level[flag & ~fr & truth]
        print(f"\n  frauds recovered by the ML layer: {len(gained)}")
        for k, v in gained[M.GROUND_TRUTH_COLUMN].value_counts().items():
            print(f"    {k:<24} {v:>3}")
    else:
        print("  isolation_forest not in this run")

    print("\n" + "=" * 62)
    print("MISSED FRAUDS BY TYPE")
    print("=" * 62)
    miss = work_level[~flag & truth]
    if len(miss):
        for k, v in miss[M.GROUND_TRUTH_COLUMN].value_counts().items():
            print(f"  {k:<26} {v:>4}")
    else:
        print("  none")


if __name__ == "__main__":
    raw = pd.read_csv(C.SYNTHETIC_CSV)
    df, scores, evidence, skipped = E.run(raw)
    report(df, scores, skipped)

    print("\n" + "=" * 62)
    print("SAMPLE AUDIT EVIDENCE — top 3 by risk")
    print("=" * 62)
    for idx in df.nlargest(3, "risk_score").index:
        r = df.loc[idx]
        print(f"\n  {r.work_id}   risk {r.risk_score}")
        print(f"  {r.work_name[:64]}")
        print(f"  {r.district} / {r.ward_or_village} / {r.implementing_agency}")
        for s in E.signals_for(df, scores, evidence, idx):
            print(f"    [{s.detector} {s.score:.2f}] {s.headline}")

    df.to_csv(C.FLAGGED_CSV, index=False)