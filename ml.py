"""
Unsupervised anomaly detection.

Why this exists
---------------
The seven rule detectors each look for one pattern we thought of in
advance. Isolation Forest looks for works that are odd without anyone
having to say what "odd" means.

It needs no labels, which matters: fraud has none. Nobody publishes a
list of confirmed fraudulent MPLADS works, so supervised learning on
fraud is impossible. Isolation Forest does not care — it only asks how
easily a record separates from the rest.

The idea in one line
--------------------
Cut the feature space with random splits until every record sits alone.
A record deep inside the crowd needs many cuts. A record off on its own
is isolated in two or three. Few cuts = anomaly.

The design decision that matters
--------------------------------
Raw amounts are useless as features. A road costs more than a borewell,
so "high amount" would just mean "this is a road". Every feature here is
therefore PEER-RELATIVE: how does this work compare to other works of the
same type in the same district? That is what makes a 0.4 km road and a
6 km road comparable.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config as C
import models as M


FEATURES = [
    "log_amount",        # scale — some fraud is simply large
    "rate_ratio",        # rate per unit vs district-category peers
    "duration_ratio",    # build time vs peers of the same type
    "paid_fraction",     # how much of the sanction has gone out
    "revision_ratio",    # revised vs sanctioned
    "ask_ratio",         # what the MP asked vs what was sanctioned
    "age_ratio",         # age vs the one-year guideline
]


def _peer_median(series, keys):
    """Median of `series` within each (district, work type) group."""
    med = series.groupby(keys).transform("median")
    return med.replace(0, np.nan)


def build_features(df):
    """Returns (feature_frame, list_of_features_actually_built)."""
    keys = [df.district, df.work_name]
    f = pd.DataFrame(index=df.index)
    built = []

    amt = df.sanctioned_amount.astype(float)
    f["log_amount"] = np.log10(amt.clip(lower=1))
    built.append("log_amount")

    if "quantity" in df.columns and df.quantity.notna().any():
        rate = amt / df.quantity.replace(0, np.nan)
    else:
        rate = amt                       # no quantity: compare raw amounts
    f["rate_ratio"] = rate / _peer_median(rate, keys)
    built.append("rate_ratio")

    if M.available(df, ["sanction_date", "completion_date"]):
        dur = (df.completion_date - df.sanction_date).dt.days
        f["duration_ratio"] = dur / _peer_median(dur, keys)
        built.append("duration_ratio")

    if M.available(df, ["expenditure", "revised_amount"]):
        f["paid_fraction"] = (df.expenditure
                              / df.revised_amount.replace(0, np.nan))
        built.append("paid_fraction")

    if M.available(df, ["revised_amount"]):
        f["revision_ratio"] = df.revised_amount / amt.replace(0, np.nan)
        built.append("revision_ratio")

    if M.available(df, ["recommended_amount"]):
        f["ask_ratio"] = df.recommended_amount / amt.replace(0, np.nan)
        built.append("ask_ratio")

    if M.available(df, ["sanction_date"]):
        age = (C.TODAY - df.sanction_date).dt.days
        f["age_ratio"] = age / C.COMPLETION_LIMIT_DAYS
        built.append("age_ratio")

    f = f[built]

    # Isolation Forest cannot take NaN. Fill with the column median and
    # record where we filled, so a missing value never looks anomalous
    # just for being missing.
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.fillna(f.median(numeric_only=True)).fillna(0.0)
    return f, built


def d_isolation_forest(df):
    """Detector eight. Same contract as the rest: (scores, evidence)."""
    if not M.available(df, ["district", "work_name", "sanctioned_amount"]):
        blank = pd.Series(0.0, index=df.index)
        ev = pd.Series([{"skipped": "needs district, work type and amount"}
                        for _ in range(len(df))], index=df.index)
        return blank, ev

    feats, built = build_features(df)

    iso = IsolationForest(
        n_estimators=C.ISO_TREES,
        contamination=C.ISO_CONTAMINATION,
        max_samples=min(C.ISO_MAX_SAMPLES, len(feats)),
        random_state=C.SEED,
        n_jobs=-1,
    )
    iso.fit(feats)

    # score_samples: closer to 0 = more anomalous. Flip and rescale so
    # higher means stranger, on the same 0-1 scale as every other detector.
    raw = -iso.score_samples(feats)
    lo, hi = np.quantile(raw, [0.50, 0.999])
    scores = pd.Series(((raw - lo) / max(hi - lo, 1e-9)).clip(0, 1),
                       index=df.index)

    # Which feature is furthest from normal? Not a true explanation —
    # Isolation Forest has none — but it tells an officer where to look.
    z = (feats - feats.median()) / feats.std(ddof=0).replace(0, np.nan)
    z = z.abs().fillna(0)
    driver = z.idxmax(axis=1)

    ev = []
    for i in df.index:
        if scores[i] <= 0:
            ev.append({})
            continue
        d = driver[i]
        ev.append({
            "anomaly_score": round(float(scores[i]), 3),
            "most_unusual_feature": d,
            "feature_value": round(float(feats.at[i, d]), 3),
            "peer_typical": round(float(feats[d].median()), 3),
            "features_used": built,
            "method": "Isolation Forest, unsupervised",
        })
    return scores, pd.Series(ev, index=df.index)


PRETTY_FEATURE = {
    "log_amount": "sanctioned amount",
    "rate_ratio": "rate per unit against district peers",
    "duration_ratio": "build time against district peers",
    "paid_fraction": "share of the sanction already paid",
    "revision_ratio": "revised against sanctioned amount",
    "ask_ratio": "amount recommended against amount sanctioned",
    "age_ratio": "age against the one-year completion limit",
}


def h_isolation_forest(e, row):
    name = PRETTY_FEATURE.get(e["most_unusual_feature"],
                              e["most_unusual_feature"])
    return (f"This combination of values does not appear elsewhere in the "
            f"register. The most unusual element is {name}: "
            f"{e['feature_value']} against a typical {e['peer_typical']}. "
            f"Flagged by unsupervised anomaly detection across "
            f"{len(e['features_used'])} measures, so no single rule "
            f"explains it \u2014 worth a human look.")