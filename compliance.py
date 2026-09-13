"""
Guideline compliance rules.

These are NOT detectors, and keeping them separate is the point.

A statistical detector says "this looks unusual". That is an opinion, and
a district officer can argue with it. A compliance rule says "this
sanction has been pending 312 days; the Guidelines allow 45". That is a
fact, and nobody argues with arithmetic.

So compliance results are reported in their own bucket. They are never
blended into the risk score, because averaging a certainty with a
probability produces a number that means neither.

Every rule below cites the clause it enforces. If the Guidelines change,
one number changes here.
"""

import numpy as np
import pandas as pd

import config as C
import models as M


# ═════════════════════════════════════════════════════════════
# THE RULES
#
# Each returns (breach: bool Series, detail: dict of Series)
# ═════════════════════════════════════════════════════════════

def r_sanction_sla(df, as_of):
    """§3.2.4 — a recommendation must be sanctioned or rejected within
    45 days of receipt."""
    if not M.available(df, ["recommendation_date"]):
        return None, "needs recommendation date"

    age = (as_of - df.recommendation_date).dt.days

    if "ida_approval" in df.columns:
        undecided = df.ida_approval.str.lower().str.strip() == "action pending"
    elif "sanction_date" in df.columns:
        # M.available() is deliberately NOT used here. It returns False for
        # an all-null column, but an all-null sanction_date is exactly the
        # case this rule exists for: nothing has been sanctioned at all.
        undecided = df.sanction_date.isna()
    else:
        return None, "needs approval status or sanction date"

    breach = (undecided & (age > C.SANCTION_SLA_DAYS)).fillna(False)
    return breach, {"days_pending": age,
                    "days_over_limit": (age - C.SANCTION_SLA_DAYS).clip(lower=0)}


def r_completion_limit(df, as_of):
    """§3.2.13 — sanctioned works are generally to be completed within
    one year of sanction."""
    if not M.available(df, ["sanction_date", "status"]):
        return None, "needs sanction date and status"

    age = (as_of - df.sanction_date).dt.days
    open_work = ~df.status.str.lower().isin(["completed", "complete"])
    breach = (open_work & (age > C.COMPLETION_LIMIT_DAYS)).fillna(False)
    return breach, {"days_since_sanction": age,
                    "days_over_limit": (age - C.COMPLETION_LIMIT_DAYS).clip(lower=0)}


def r_minimum_sanction(df, as_of=None):
    """§3.2.9 — the amount sanctioned for an individual work shall
    NORMALLY be not less than Rs 2.5 lakh.

    The word is 'normally'. A lower amount is permitted for reasons
    recorded in the sanction letter, and those letters are not public.
    So this is not a violation count — it is a count of works that
    require a recorded justification.
    """
    if not M.available(df, ["sanctioned_amount"]):
        return None, "needs sanctioned amount"

    amt = df.sanctioned_amount.astype(float)
    breach = (amt < C.MIN_SANCTION) & (amt > 0)
    return breach.fillna(False), {"shortfall": (C.MIN_SANCTION - amt).clip(lower=0)}


RULES = {
    "sanction_sla": {
        "fn": r_sanction_sla,
        "clause": "\u00a73.2.4",
        "title": "Sanction decision beyond 45 days",
        "severity": "VIOLATION",
        "text": ("Recommended {days_pending:.0f} days ago and still awaiting a "
                 "decision. Guidelines \u00a73.2.4 require sanction or rejection "
                 "within {limit} days \u2014 {days_over_limit:.0f} days over."),
    },
    "completion_limit": {
        "fn": r_completion_limit,
        "clause": "\u00a73.2.13",
        "title": "Incomplete beyond one year of sanction",
        "severity": "VIOLATION",
        "text": ("Sanctioned {days_since_sanction:.0f} days ago and not marked "
                 "complete. Guidelines \u00a73.2.13 set a one-year limit \u2014 "
                 "{days_over_limit:.0f} days over."),
    },
    "minimum_sanction": {
        "fn": r_minimum_sanction,
        "clause": "\u00a73.2.9",
        "title": "Below the Rs 2.5 lakh sanction floor",
        "severity": "NEEDS JUSTIFICATION",
        "text": ("Sanctioned at Rs {amount_lakh:.2f} lakh. Guidelines \u00a73.2.9 "
                 "say works shall normally be not less than Rs 2.5 lakh, and "
                 "permit less only for reasons recorded in the sanction letter. "
                 "Retrieve that justification."),
    },
}


# ═════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════

def snapshot_date(df):
    """The date this data is current as of.

    This matters more than it looks. A scraped export is a snapshot, not
    a live feed. Measuring a 2024 snapshot against today's date would
    report every pending work as two years overdue, which says more
    about when we downloaded the file than about the ministry.
    """
    dates = [df[c].max() for c in ("recommendation_date", "sanction_date",
                                   "completion_date") if c in df.columns]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else C.TODAY


def check(df, as_of=None):
    """Returns (breach_frame, notes_per_work, skipped, as_of).

    as_of : the date to measure against. Defaults to the newest date in
            the data, so a snapshot is judged as of when it was taken.
    """
    if as_of is None:
        as_of = snapshot_date(df)

    breaches = pd.DataFrame(index=df.index)
    details, skipped = {}, []

    for name, spec in RULES.items():
        out, extra = spec["fn"](df, as_of)
        if out is None:
            skipped.append((name, extra))
            breaches[name] = False
            continue
        breaches[name] = out
        details[name] = extra

    amt_lakh = df.sanctioned_amount.astype(float) / 1e5
    notes = []
    for i in df.index:
        row = []
        for name, spec in RULES.items():
            if not breaches.at[i, name]:
                continue
            ctx = {"limit": C.SANCTION_SLA_DAYS, "amount_lakh": amt_lakh[i]}
            for k, series in details.get(name, {}).items():
                ctx[k] = series[i]
            row.append({
                "rule": name,
                "clause": spec["clause"],
                "title": spec["title"],
                "severity": spec["severity"],
                "text": spec["text"].format(**ctx),
            })
        notes.append(row)

    return breaches, pd.Series(notes, index=df.index), skipped, as_of


def summary(df, breaches):
    """Headline counts and rupee values, per rule."""
    amt = df.sanctioned_amount.astype(float)
    rows = []
    for name, spec in RULES.items():
        b = breaches[name]
        rows.append({
            "Clause": spec["clause"],
            "Rule": spec["title"],
            "Severity": spec["severity"],
            "Works": int(b.sum()),
            "Share": b.mean(),
            "Value (Rs cr)": round(float(amt[b].sum()) / 1e7, 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    import adapters as A

    if len(sys.argv) > 1 and sys.argv[1] == "real":
        raw = pd.read_csv(C.REAL_CSV, sep=";", dtype=str)
        df, meta = A.from_esakshi_export(raw)
        label = "REAL eSAKSHI EXPORT"
    else:
        df = pd.read_csv(C.SYNTHETIC_CSV)
        for c in M.DATE_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        label = "SYNTHETIC REGISTER"

    breaches, notes, skipped, as_of = check(df)

    print("=" * 70)
    print(f"GUIDELINE COMPLIANCE  \u2014  {label}")
    print(f"{len(df):,} works  \u00b7  measured as of {as_of.date()} "
          f"(newest date in the data)")
    print("=" * 70)
    print(summary(df, breaches).to_string(index=False))

    if skipped:
        print("\n  rules that could not run on this source:")
        for n, why in skipped:
            print(f"    {n}: {why}")

    any_breach = breaches.any(axis=1)
    print(f"\n  works breaching at least one rule: {int(any_breach.sum()):,} "
          f"({any_breach.mean():.1%})")

    ex = df[any_breach]
    if len(ex):
        i = ex.index[0]
        print("\n  example \u2014", df.at[i, "work_id"])
        for n in notes[i]:
            print(f"    [{n['severity']}] {n['clause']}  {n['text']}")