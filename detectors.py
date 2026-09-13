"""
The detectors.

Every detector has the same shape:

    in  : the whole works table
    out : (scores, evidence)
          scores   = one 0-1 number per row
          evidence = one dict per row, holding the raw numbers
                     behind that score

Each detector declares REQUIRES. If those columns are missing from
the data source, the detector skips itself and returns zeros with a
note saying why. That is how one engine runs on both the synthetic
register and the real eSAKSHI scrape.
"""

import re

import numpy as np
import pandas as pd

import config as C
import models as M


# ═════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════

def _mad_z(series):
    """Robust z-score.

    Median absolute deviation resists the outliers we are hunting.
    A standard deviation does not: one huge value inflates the SD
    and then hides inside the wider band it created.
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - med) / mad


_WORD = re.compile(r"[a-z0-9]+")

# Words that appear in almost every description and therefore
# carry no information about WHICH work this is.
_STOPWORDS = {
    "construction", "of", "at", "in", "the", "and", "for", "near",
    "providing", "installation", "development", "building", "works",
    "work", "along", "road", "new", "a", "to", "with",
}


def _tokens(text):
    """Turn a description into a set of meaningful words."""
    if not isinstance(text, str) or not text.strip():
        return set()
    words = _WORD.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _text_similarity(a, b):
    """Jaccard overlap: shared words / total distinct words.

    1.0 = identical wording
    0.0 = nothing in common

    Returns None when either side has no usable text, because
    "no description" is not the same as "different description".
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return None
    return len(ta & tb) / len(ta | tb)


def _blank(df):
    """A zero score and an empty evidence dict for every row."""
    return (pd.Series(0.0, index=df.index),
            pd.Series([{} for _ in range(len(df))], index=df.index))


def _skip(df, reason):
    scores, ev = _blank(df)
    ev = pd.Series([{"skipped": reason} for _ in range(len(df))],
                   index=df.index)
    return scores, ev


# ═════════════════════════════════════════════════════════════
# 1. GHOST WORK
# ═════════════════════════════════════════════════════════════

def d_ghost_work(df):
    """Marked complete and paid, but built impossibly fast.

    Measured from SANCTION date, not work-start date. Sanction date
    is set by the District Authority; work-start is entered by the
    agency being investigated. And the real eSAKSHI export has no
    work-start date at all.
    """
    REQ = ["sanction_date", "completion_date", "expenditure",
           "revised_amount", "status"]
    if not M.available(df, REQ):
        return _skip(df, "needs completion date and expenditure")

    days = (df.completion_date - df.sanction_date).dt.days
    paid = df.expenditure / df.revised_amount.replace(0, np.nan)
    done = df.status.str.lower().isin(["completed", "complete"])

    hit = (days < C.GHOST_MAX_DAYS) & (paid > C.GHOST_MIN_PAID) & done
    hit = hit.fillna(False)

    # peer benchmark: how long this kind of work normally takes here
    peer_days = days.groupby([df.district, df.work_name]).transform("median")

    scores = hit.astype(float)
    ev = []
    for i in df.index:
        if not hit[i]:
            ev.append({})
            continue
        ev.append({
            "days_sanction_to_completion": int(days[i]),
            "peer_median_days": (None if pd.isna(peer_days[i])
                                 else float(peer_days[i])),
            "amount_paid": float(df.expenditure[i]),
            "amount_sanctioned": float(df.revised_amount[i]),
            "paid_fraction": round(float(paid[i]), 3),
        })
    return scores, pd.Series(ev, index=df.index)


def h_ghost_work(e, row):
    peer = e.get("peer_median_days")
    tail = (f" Comparable works in this district take about "
            f"{peer:.0f} days." if peer else "")
    return (f"Marked complete and {e['paid_fraction']*100:.0f}% paid "
            f"{e['days_sanction_to_completion']} days after sanction."
            + tail)


# ═════════════════════════════════════════════════════════════
# 2. DUPLICATE
# ═════════════════════════════════════════════════════════════

def d_duplicate(df):
    """One work, two sanctions.

    v1 matched on district + work type + agency + amount + date.
    That also describes two genuine works, which is why it produced
    214 of 221 false positives.

    v2 adds a location test, in order of preference:
      1. description text overlap   (best)
      2. same ward or village       (fallback)
      3. neither available          (heavily discounted)
    """
    REQ = ["district", "work_name", "implementing_agency",
           "sanctioned_amount", "recommendation_date"]
    if not M.available(df, REQ):
        return _skip(df, "needs district, work type, agency, amount, date")

    date_col = ("sanction_date"
                if M.available(df, ["sanction_date"])
                else "recommendation_date")

    scores = pd.Series(0.0, index=df.index)
    ev = {i: {} for i in df.index}

    key = ["district", "work_name", "implementing_agency"]
    for _, grp in df.groupby(key, sort=False):
        if len(grp) < 2:
            continue
        g = grp.sort_values(date_col)
        amts = g.sanctioned_amount.values
        dates = g[date_col].values
        descs = g.work_description.values if "work_description" in g else [""] * len(g)
        places = (g.ward_or_village.values
                  if "ward_or_village" in g else [""] * len(g))
        idxs = g.index.values

        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                gap = abs((dates[b] - dates[a]) / np.timedelta64(1, "D"))
                if gap > C.DUP_WINDOW_DAYS:
                    break                      # sorted, so nothing later fits
                base = max(amts[a], 1)
                rel = abs(amts[b] - amts[a]) / base
                if rel > C.DUP_AMOUNT_TOL:
                    continue

                # amount + date match. Now: same PLACE?
                sim = _text_similarity(descs[a], descs[b])
                same_place = (isinstance(places[a], str)
                              and places[a] == places[b]
                              and places[a] != "")

                if sim is not None:
                    if sim < C.DUP_MIN_TEXT_SIM:
                        continue               # different places, not a dupe
                    conf, basis = 1.00, "description"
                elif same_place:
                    # weaker evidence, so demand a closer match
                    if rel > C.DUP_FALLBACK_TOL or gap > C.DUP_FALLBACK_WINDOW:
                        continue
                    conf, basis = 0.70, "ward_or_village"
                else:
                    conf, basis = 0.35, "amount_and_date_only"

                s = (1 - rel * 10) * conf
                for x, y in ((a, b), (b, a)):
                    if s > scores[idxs[x]]:
                        scores[idxs[x]] = s
                        ev[idxs[x]] = {
                            "match_work_id": g.work_id.values[y],
                            "days_apart": int(gap),
                            "amount_this": float(amts[x]),
                            "amount_other": float(amts[y]),
                            "amount_difference_pct": round(rel * 100, 2),
                            "text_similarity": (None if sim is None
                                                else round(sim, 3)),
                            "match_basis": basis,
                            "confidence": conf,
                        }

    return scores.clip(0, 1), pd.Series([ev[i] for i in df.index],
                                        index=df.index)


def h_duplicate(e, row):
    if e["match_basis"] == "description":
        how = (f"Descriptions are {e['text_similarity']*100:.0f}% identical.")
    elif e["match_basis"] == "ward_or_village":
        how = "Same ward/village, but no description to compare."
    else:
        how = "No description or location to confirm; amount and date only."
    return (f"Near-identical sanction {e['days_apart']} days apart in the "
            f"same district by the same agency. Amounts differ by "
            f"{e['amount_difference_pct']:.1f}%. {how} "
            f"Matching work: {e['match_work_id']}.")


# ═════════════════════════════════════════════════════════════
# 3. UNIT COST
# ═════════════════════════════════════════════════════════════

def d_unit_cost(df):
    """Rate per unit far from what neighbours pay.

    If quantity is available we compare true rates. The real
    eSAKSHI export has no quantity, so we fall back to comparing
    raw sanctioned amounts within the same work type and district.
    Weaker, but a borewell is roughly a borewell.
    """
    REQ = ["district", "work_name", "sanctioned_amount"]
    if not M.available(df, REQ):
        return _skip(df, "needs district, work type and amount")

    has_qty = M.available(df, ["quantity"])
    if has_qty:
        value = df.sanctioned_amount / df.quantity.replace(0, np.nan)
        basis = "rate_per_unit"
    else:
        value = df.sanctioned_amount.astype(float)
        basis = "sanctioned_amount"

    grp = [df.district, df.work_name]
    z = value.groupby(grp).transform(_mad_z)
    med = value.groupby(grp).transform("median")
    n = value.groupby(grp).transform("size")

    scores = ((z.abs() - C.UNIT_COST_Z_FLOOR) / C.UNIT_COST_Z_RANGE)
    scores = scores.clip(0, 1).fillna(0)
    scores[n < C.UNIT_COST_MIN_PEERS] = 0.0     # too few peers to judge

    ev = []
    for i in df.index:
        if scores[i] <= 0:
            ev.append({})
            continue
        ev.append({
            "basis": basis,
            "value": float(value[i]),
            "peer_median": float(med[i]),
            "peer_count": int(n[i]),
            "modified_z": round(float(z[i]), 2),
            "ratio_to_median": (round(float(value[i] / med[i]), 2)
                                if med[i] else None),
            "unit": (row_unit(df, i) if has_qty else None),
        })
    return scores, pd.Series(ev, index=df.index)


def row_unit(df, i):
    return df.unit[i] if "unit" in df.columns else None


def h_unit_cost(e, row):
    per = f" per {e['unit']}" if e.get("unit") else ""
    direction = "above" if e["ratio_to_median"] >= 1 else "below"
    return (f"Rs {e['value']/1e5:.1f} lakh{per}, against a district median of "
            f"Rs {e['peer_median']/1e5:.1f} lakh across {e['peer_count']} "
            f"comparable works. {e['ratio_to_median']}x the peer rate "
            f"({direction} normal).")


# ═════════════════════════════════════════════════════════════
# 4. COST OVERRUN
# ═════════════════════════════════════════════════════════════

def d_cost_overrun(df):
    """Price grew after approval with no recorded scope change."""
    REQ = ["revised_amount", "sanctioned_amount"]
    if not M.available(df, REQ):
        return _skip(df, "needs both sanctioned and revised amount")

    ratio = df.revised_amount / df.sanctioned_amount.replace(0, np.nan)
    scores = ((ratio - C.OVERRUN_FLOOR) / C.OVERRUN_RANGE).clip(0, 1).fillna(0)

    ev = []
    for i in df.index:
        if scores[i] <= 0:
            ev.append({})
            continue
        ev.append({
            "sanctioned_amount": float(df.sanctioned_amount[i]),
            "revised_amount": float(df.revised_amount[i]),
            "increase_amount": float(df.revised_amount[i] - df.sanctioned_amount[i]),
            "ratio": round(float(ratio[i]), 2),
        })
    return scores, pd.Series(ev, index=df.index)


def h_cost_overrun(e, row):
    return (f"Sanctioned at Rs {e['sanctioned_amount']/1e5:.1f} lakh, revised "
            f"to Rs {e['revised_amount']/1e5:.1f} lakh. That is "
            f"{e['ratio']}x the original, an increase of "
            f"Rs {e['increase_amount']/1e5:.1f} lakh with no scope change "
            f"recorded.")


# ═════════════════════════════════════════════════════════════
# 5. STALLED WORK
# ═════════════════════════════════════════════════════════════

def d_stalled_work(df):
    """Sanctioned long ago, no vendor payment ever raised.

    Renamed from "idle funds". Under the TSA/Hybrid fund flow
    (April 2025) money is released to vendors just-in-time, so there
    is no parked cash to sit idle. What the data actually shows is a
    sanction that never converted into work.
    """
    REQ = ["sanction_date", "expenditure", "status"]
    if not M.available(df, REQ):
        return _skip(df, "needs sanction date and expenditure")

    age = (C.TODAY - df.sanction_date).dt.days
    open_work = ~df.status.str.lower().isin(["completed", "complete"])
    nothing_paid = df.expenditure.fillna(0) <= 0

    overdue = (age - C.IDLE_MIN_AGE_DAYS).clip(lower=0)
    scores = ((open_work & nothing_paid) *
              (overdue / C.IDLE_AGE_RANGE)).clip(0, 1).fillna(0)

    ev = []
    for i in df.index:
        if scores[i] <= 0:
            ev.append({})
            continue
        ev.append({
            "days_since_sanction": int(age[i]),
            "days_past_one_year_limit": int(age[i] - C.COMPLETION_LIMIT_DAYS),
            "amount_sanctioned": float(df.sanctioned_amount[i]),
            "amount_paid": float(df.expenditure.fillna(0)[i]),
        })
    return scores, pd.Series(ev, index=df.index)


def h_stalled_work(e, row):
    return (f"Sanctioned {e['days_since_sanction']} days ago. No vendor "
            f"payment has ever been raised and the work is not complete. "
            f"Rs {e['amount_sanctioned']/1e5:.1f} lakh committed, "
            f"{e['days_past_one_year_limit']} days past the one-year "
            f"completion guideline.")


# ═════════════════════════════════════════════════════════════
# 6. ROUND NUMBER
# ═════════════════════════════════════════════════════════════

def d_round_number(df):
    """Suspiciously tidy amounts.

    CORROBORATION ONLY. Never enough on its own.

    On real MPLADS data 25% of amounts are exact multiples of
    Rs 5 lakh, so this fires constantly on honest works. Its weight
    is set so it cannot cross the risk threshold by itself.
    """
    if not M.available(df, ["sanctioned_amount"]):
        return _skip(df, "needs sanctioned amount")

    a = df.sanctioned_amount.fillna(0)
    scores = pd.Series(0.0, index=df.index)
    # Capped at 0.80 on purpose. 0.80 x weight 0.45 = 0.36, which is below
    # RISK_THRESHOLD (0.40), so this detector can never surface a work on its
    # own. On real MPLADS data a quarter of all amounts are exact multiples
    # of Rs 5 lakh, so left uncapped it would flag thousands of honest works.
    scores[a % 10_000 == 0] = 0.30
    scores[a % 100_000 == 0] = 0.50
    scores[a % 500_000 == 0] = 0.65
    scores[a % 1_000_000 == 0] = 0.80
    scores[a <= 0] = 0.0

    ev = []
    for i in df.index:
        if scores[i] <= 0:
            ev.append({})
            continue
        amt = float(a[i])
        step = (1_000_000 if amt % 1_000_000 == 0 else
                500_000 if amt % 500_000 == 0 else
                100_000 if amt % 100_000 == 0 else 10_000)
        ev.append({"amount": amt, "round_to": step,
                   "note": "corroborating signal only"})
    return scores, pd.Series(ev, index=df.index)


def h_round_number(e, row):
    return (f"Amount is exactly Rs {e['amount']/1e5:.1f} lakh, a round "
            f"multiple of Rs {e['round_to']/1e5:.1f} lakh. Weak on its own; "
            f"treat only as supporting evidence.")


# ═════════════════════════════════════════════════════════════
# 7. AGENCY CAPTURE
# ═════════════════════════════════════════════════════════════

def d_agency_capture(df):
    """One agency holds an unusual share of a district's works.

    This scores a DISTRICT, not a work. It does not say a work is
    bad; it says this is a place worth looking at.
    """
    REQ = ["district", "implementing_agency"]
    if not M.available(df, REQ):
        return _skip(df, "needs district and implementing agency")

    scores = pd.Series(0.0, index=df.index)
    ev = {i: {} for i in df.index}

    for district, grp in df.groupby("district"):
        share = grp.implementing_agency.value_counts(normalize=True)
        hhi = float((share ** 2).sum())
        if hhi <= C.HHI_FLOOR:
            continue
        dominant = share.index[0]
        s = min((hhi - C.HHI_FLOOR) / C.HHI_RANGE, 1.0)
        hit = grp.index[grp.implementing_agency == dominant]
        for i in hit:
            scores[i] = s
            ev[i] = {
                "district": district,
                "dominant_agency": dominant,
                "dominant_share": round(float(share.iloc[0]), 3),
                "hhi": round(hhi, 3),
                "agency_count": int(len(share)),
                "district_work_count": int(len(grp)),
            }
    return scores, pd.Series([ev[i] for i in df.index], index=df.index)


def h_agency_capture(e, row):
    return (f"{e['dominant_agency']} holds "
            f"{e['dominant_share']*100:.0f}% of the "
            f"{e['district_work_count']} works in {e['district']} "
            f"(concentration index {e['hhi']}, {e['agency_count']} agencies "
            f"active). A risk condition, not a finding on its own.")


# ═════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════

import ml  # noqa: E402  (imported here to keep the detector list readable)

DETECTORS = {
    "ghost_work":     (d_ghost_work,     h_ghost_work),
    "duplicate":      (d_duplicate,      h_duplicate),
    "unit_cost":      (d_unit_cost,      h_unit_cost),
    "cost_overrun":   (d_cost_overrun,   h_cost_overrun),
    "stalled_work":   (d_stalled_work,   h_stalled_work),
    "round_number":   (d_round_number,   h_round_number),
    "agency_capture": (d_agency_capture, h_agency_capture),
    "isolation_forest": (ml.d_isolation_forest, ml.h_isolation_forest),
}