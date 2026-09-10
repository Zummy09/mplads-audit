"""
Adapters.

One job: take whatever shape a data source arrives in, and produce the
standard column set the detectors expect. Nothing downstream changes.

    real eSAKSHI scrape  ──┐
                           ├──► standard schema ──► detectors
    synthetic register   ──┘

Each adapter also returns:
  missing   — standard columns this source simply does not carry
  unreliable— detectors that would produce noise on this source, and why

That second list is what stops the engine from quietly reporting
nonsense. A detector that cannot work here says so, out loud.
"""

import numpy as np
import pandas as pd

import models as M


# ═════════════════════════════════════════════════════════════
# SYNTHETIC REGISTER  (our generator)
# ═════════════════════════════════════════════════════════════

def from_synthetic(df):
    """Already in the standard shape. Here for symmetry."""
    return df.copy(), {"missing": [], "unreliable": {},
                       "source": "synthetic register"}


# ═════════════════════════════════════════════════════════════
# REAL eSAKSHI WORKS EXPORT
#
# 15 columns, semicolon-separated:
#   MP NAME · WORK · CATEGORY · STATE · CONSTITUENCY · IDA
#   CITY · WARD · BLOCK · VILLAGE · RECOMMENDED DATE
#   ALLOCATION AMOUNT · IDA APPROVAL · STATUS · HOUSE
# ═════════════════════════════════════════════════════════════

STATUS_MAP = {
    "completed": "Completed",
    "ongoing": "In Progress",
    "sanctioned": "In Progress",
    "unsanctioned": "Unsanctioned",
}


def _split_work(value):
    """'WS/MP300/2023-2024/352 - Street lights' -> (work_no, activity).

    Unsanctioned works have no work number yet and arrive as 'NA - ...'.
    """
    if not isinstance(value, str) or " - " not in value:
        return None, (value if isinstance(value, str) else "")
    head, tail = value.split(" - ", 1)
    head = head.strip()
    return (None if head.upper() == "NA" else head), tail.strip()


def from_esakshi_export(df):
    """df = pd.read_csv(path, sep=';')"""
    src = df.copy()
    src.columns = [c.strip().upper() for c in src.columns]

    parsed = src["WORK"].apply(_split_work)
    work_no = parsed.apply(lambda t: t[0])
    activity = parsed.apply(lambda t: t[1])

    rural = src["VILLAGE"].fillna("").str.strip().ne("")
    ward_or_village = np.where(rural,
                               src["VILLAGE"].fillna("").str.strip(),
                               src["WARD"].fillna("").str.strip())

    amount = pd.to_numeric(src["ALLOCATION AMOUNT"], errors="coerce")

    out = pd.DataFrame({
        # identity
        "work_id": work_no.fillna(
            pd.Series([f"UNSANCTIONED/{i:06d}" for i in range(len(src))],
                      index=src.index)),
        "state": src["STATE"].fillna("").str.strip(),
        # Peer comparison uses the constituency, not the district: it is the
        # only clean, fully-populated geographic key in this export, and for
        # MPLADS it is the natural unit anyway — one MP, one entitlement.
        "district": src["CONSTITUENCY"].fillna("").str.strip(),
        "constituency": src["CONSTITUENCY"].fillna("").str.strip(),
        "mp_name": src["MP NAME"].fillna("").str.strip(),

        # what was built
        "work_name": activity,
        "work_category": src["CATEGORY"].fillna("").str.strip(),
        "work_description": "",          # not present in this export
        "location_type": np.where(rural, "Rural", "Urban"),
        "ward_or_village": ward_or_village,
        "block": src["BLOCK"].fillna("").str.strip(),

        # who
        "implementing_agency": src["IDA"].fillna("").str.strip(),
        "ida_approval": src["IDA APPROVAL"].fillna("").str.strip(),

        # dates — only one is published
        "recommendation_date": pd.to_datetime(src["RECOMMENDED DATE"],
                                              errors="coerce"),
        "sanction_date": pd.NaT,
        "work_start_date": pd.NaT,
        "completion_date": pd.NaT,

        # money — only the recommended/allocated amount is published
        "recommended_amount": amount,
        "sanctioned_amount": amount,
        "revised_amount": np.nan,
        "expenditure": np.nan,

        "status": src["STATUS"].fillna("").str.strip().str.lower()
                     .map(STATUS_MAP).fillna("Unsanctioned"),
        "house": src["HOUSE"].fillna("").str.strip(),
    })

    # drop rows with no usable amount at all
    out = out[out.sanctioned_amount.notna() & (out.sanctioned_amount > 0)]
    out = out.reset_index(drop=True)

    meta = {
        "source": "eSAKSHI works export",
        "rows_in": len(src),
        "rows_out": len(out),
        "missing": [
            "sanction_date", "work_start_date", "completion_date",
            "revised_amount", "expenditure", "quantity", "unit",
            "work_description",
        ],
        "unreliable": {
            "agency_capture":
                "IDA in this export is the District Authority, not the "
                "executing agency. One district has one Collector, so "
                "concentration is 100% everywhere and means nothing.",
        },
        "notes": [
            "ALLOCATION AMOUNT is what the MP recommended. The sanctioned "
            "and paid amounts are not published, so unit-cost comparison "
            "runs on recommended value.",
            "No quantity column, so unit cost compares amounts within the "
            "same activity and constituency rather than a true rate.",
            "Duplicate matching falls back to ward/village, because this "
            "export carries no work description.",
        ],
    }
    return out, meta


# ═════════════════════════════════════════════════════════════
# COMPLIANCE CHECKS THAT ONLY REAL DATA CAN ANSWER
#
# These are not statistical detectors. They are counts of breaches of
# rules the Guidelines state in writing.
# ═════════════════════════════════════════════════════════════

def compliance_summary(std, today=None, sla_days=45, min_sanction=250_000):
    """Returns a dict of countable Guideline breaches."""
    today = today or std.recommendation_date.max()
    age = (today - std.recommendation_date).dt.days

    pending = std.ida_approval.str.lower().eq("action pending") \
        if "ida_approval" in std.columns else pd.Series(False, index=std.index)

    over_sla = pending & (age > sla_days)
    below_min = std.sanctioned_amount < min_sanction

    return {
        "as_of": today,
        "works": len(std),
        "total_value": float(std.sanctioned_amount.sum()),
        "pending_decision": int(pending.sum()),
        "past_45_day_sla": int(over_sla.sum()),
        "past_45_day_value": float(std.loc[over_sla, "sanctioned_amount"].sum()),
        "below_min_sanction": int(below_min.sum()),
        "below_min_value": float(std.loc[below_min, "sanctioned_amount"].sum()),
        "status_mix": std.status.value_counts().to_dict(),
    }


# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import engine as E

    path = sys.argv[1] if len(sys.argv) > 1 else "data/mplads_real.csv"
    raw = pd.read_csv(path, sep=";", dtype=str)
    std, meta = from_esakshi_export(raw)

    print("=" * 64)
    print(f"ADAPTER: {meta['source']}")
    print("=" * 64)
    print(f"  rows in  {meta['rows_in']:,}   rows out  {meta['rows_out']:,}")
    print(f"  columns this source does not carry:")
    print(f"    {', '.join(meta['missing'])}")
    for d, why in meta["unreliable"].items():
        print(f"  disabled: {d}\n    {why}")

    c = compliance_summary(std)
    print("\n" + "=" * 64)
    print(f"GUIDELINE COMPLIANCE  (as of {c['as_of'].date()})")
    print("=" * 64)
    print(f"  works                     {c['works']:>8,}   "
          f"Rs {c['total_value']/1e7:>8,.0f} cr")
    print(f"  awaiting a decision       {c['pending_decision']:>8,}")
    print(f"  past the 45-day limit     {c['past_45_day_sla']:>8,}   "
          f"Rs {c['past_45_day_value']/1e7:>8,.0f} cr   [Guidelines 3.2.4]")
    print(f"  below Rs 2.5 lakh floor   {c['below_min_sanction']:>8,}   "
          f"Rs {c['below_min_value']/1e7:>8,.0f} cr   [Guidelines 3.2.9]")
    print("\n  status mix:")
    for k, v in c["status_mix"].items():
        print(f"    {k:<16} {v:>8,}  ({v/c['works']:.1%})")

    print("\n" + "=" * 64)
    print("DETECTION ENGINE ON REAL DATA")
    print("=" * 64)
    df, scores, evidence, skipped = E.run(
        std, disable=list(meta["unreliable"]))
    for n, why in skipped:
        print(f"  skipped {n}: {why}")
    import config as C
    flag = df.risk_score > C.RISK_THRESHOLD
    print(f"\n  surfaced for review  {int(flag.sum()):,} of {len(df):,} works")
    print(f"  value surfaced       Rs {df.loc[flag,'sanctioned_amount'].sum()/1e7:,.0f} cr")
    print("\n  driving signal:")
    for k, v in df[flag].primary_detector.value_counts().items():
        print(f"    {k:<18} {v:>7,}")
