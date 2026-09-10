"""
The engine. Data in, flagged works out.

    df  →  validate  →  run 7 detectors  →  combine scores  →  results

Knows nothing about MPLADS, CSVs, or dashboards. It takes a table
with the standard columns and gives back scores plus evidence.
"""

import pandas as pd

import config as C
import models as M
import detectors as D
import scoring as S


def run(df, disable=None):
    """Returns (df_with_scores, scores, evidence, skipped).

    disable: detector names an adapter has declared unreliable for this
    data source. They return zeros and say why, instead of adding noise.
    """
    disable = set(disable or [])
    df = df.copy()
    M.validate(df)

    for col in M.DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "work_description" not in df.columns:
        df["work_description"] = ""
    df["work_description"] = df["work_description"].fillna("")

    scores = pd.DataFrame(index=df.index)
    evidence = {}
    skipped = []

    for name, (fn, _headline) in D.DETECTORS.items():
        if name in disable:
            s = pd.Series(0.0, index=df.index)
            ev = pd.Series([{"skipped": "not meaningful on this data source"}
                            for _ in range(len(df))], index=df.index)
        else:
            s, ev = fn(df)
        scores[name] = s.astype(float)
        evidence[name] = ev
        first = ev.iloc[0] if len(ev) else {}
        if isinstance(first, dict) and "skipped" in first:
            skipped.append((name, first["skipped"]))

    df["risk_score"], weighted = S.combine(scores)
    df["primary_detector"] = weighted.idxmax(axis=1)
    df["triggers"] = scores.apply(
        lambda r: "|".join(sorted(k for k, v in r.items()
                                  if v > C.FIRE_THRESHOLD)), axis=1)
    df["flagged"] = df.risk_score > C.RISK_THRESHOLD

    return df, scores, evidence, skipped


def signals_for(df, scores, evidence, idx):
    """Build the Signal list for one work — what the auditor reads."""
    out = []
    row = df.loc[idx]
    for name, (_fn, headline) in D.DETECTORS.items():
        sc = float(scores.at[idx, name])
        ev = evidence[name].at[idx]
        if sc <= 0 or not ev or "skipped" in ev:
            continue
        out.append(M.Signal(
            detector=name,
            score=round(sc, 3),
            fired=sc > C.FIRE_THRESHOLD,
            headline=headline(ev, row),
            evidence=ev,
        ))
    return sorted(out, key=lambda s: s.score, reverse=True)
