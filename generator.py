"""
MPLADS synthetic works register.

Builds a realistic works dataset shaped like the eSAKSHI
"Recommended Work Details" report export, and plants seven known
fraud/inefficiency patterns with a ground-truth label column so
detector recall and precision are measurable.

The detectors NEVER see fraud_label. It is used only at scoring time.
"""

import random
from datetime import timedelta

import numpy as np
import pandas as pd

import config as C
import reference as R

RNG = random.Random(C.SEED)
np.random.seed(C.SEED)


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def _amount(code, qty):
    """Sanctioned amount with honest market noise, respecting the
    Rs 2.5 lakh minimum from Guidelines para 3.2.9."""
    rate = R.WORKS[code][4]
    amt = round(rate * qty * RNG.uniform(0.88, 1.14), -3)
    return max(amt, C.MIN_SANCTION)


def _place():
    """A location phrase that goes inside a work description."""
    return f"{RNG.choice(R.LANDMARKS)}"


def _description(code, block, has_description):
    """The eSAKSHI Work Description field.

    OPTIONAL in the real system, so we leave ~30% blank. That gap is
    not an oversight — it is the condition the duplicate detector has
    to survive on real data.
    """
    if not has_description:
        return ""
    template = RNG.choice(R.DESCRIPTION_TEMPLATES[code])
    return template.format(place=_place(), area=block)


def _base_row(i, state, district, constituency, mp, blocks, agency_pool):
    code = RNG.choice(list(R.WORKS))
    parent, activity, annex, unit, rate, (lo, hi) = R.WORKS[code]

    qty = round(RNG.uniform(lo, hi), 2)
    sanctioned = _amount(code, qty)

    # MPs usually ask for roughly what gets sanctioned
    recommended = round(sanctioned * RNG.uniform(0.92, 1.06), -3)

    block = RNG.choice(blocks)
    location_type = "Rural" if RNG.random() < 0.72 else "Urban"
    ward_village = block if location_type == "Rural" else f"Ward No. {RNG.randint(1, 40)}"

    has_desc = RNG.random() < C.DESCRIPTION_COVERAGE
    agency_used = RNG.choice(agency_pool)

    # ── dates ────────────────────────────────────────────────
    rec = C.FY_START + timedelta(days=RNG.randint(0, 850))
    san = rec + timedelta(days=RNG.randint(8, 70))
    start = san + timedelta(days=RNG.randint(5, 60))

    # Duration is driven, not random. See reference.py for why.
    dur = R.BASE_DURATION.get(code, 180)
    dur *= R.AGENCY_SPEED.get(agency_used, 1.0)
    dur *= R.DISTRICT_FACTOR.get(district, 1.0)
    if san.month in R.MONSOON_MONTHS:
        dur *= R.MONSOON_FACTOR
    # bigger works take proportionally longer
    dur *= 1 + R.SIZE_FACTOR_PER_LOG * max(
        0.0, (sanctioned / 1e6) ** 0.5 - 1)
    dur = int(dur * RNG.uniform(0.72, 1.34))   # honest execution noise
    comp = start + timedelta(days=max(dur, 20))

    done = comp <= C.TODAY

    return {
        # identity
        "work_id":              f"WS/MP{i:05d}/2023-2026/{i:06d}",
        "state":                state,
        "district":             district,
        "constituency":         constituency,
        "mp_name":              mp,

        # what was built  (Annexure-VIII vocabulary)
        "pfms_code":            code,
        "work_category":        parent,
        "work_name":            activity,
        "annexure_ref":         annex,
        "work_description":     _description(code, block, has_desc),
        "unit":                 unit,
        "quantity":             qty,

        # where
        "location_type":        location_type,
        "ward_or_village":      ward_village,

        # who
        "implementing_agency":  agency_used,
        "vendor_name":          RNG.choice(R.VENDORS),

        # dates
        "recommendation_date":  rec,
        "sanction_date":        san,
        "work_start_date":      start,
        "completion_date":      comp if done else pd.NaT,

        # money
        "recommended_amount":   recommended,
        "sanctioned_amount":    sanctioned,
        "revised_amount":       sanctioned,
        "expenditure": (round(sanctioned * RNG.uniform(0.90, 1.0), -3) if done
                        else round(sanctioned * RNG.uniform(0.05, 0.70), -3)),
        "status":               "Completed" if done else "In Progress",
        "sc_st_component":      RNG.random() < 0.22,

        # ground truth — detectors never read this
        "fraud_label":          [],
    }


# ═════════════════════════════════════════════════════════════
# BUILD
# ═════════════════════════════════════════════════════════════

def build(n=C.N_BASE_WORKS):
    rows = []
    keys, mps, pools = [], {}, {}
    i = 1

    for state, districts in R.GEOGRAPHY.items():
        for district, (constituency, blocks) in districts.items():
            keys.append((state, district))
            mps[(state, district)] = f"Shri/Smt MP {constituency.title()}"
            # each district has its own slightly-skewed agency pool
            pools[(state, district)] = RNG.sample(R.AGENCIES, RNG.randint(4, 6))

    for _ in range(n):
        state, district = RNG.choice(keys)
        constituency, blocks = R.GEOGRAPHY[state][district]
        rows.append(_base_row(i, state, district, constituency,
                              mps[(state, district)], blocks,
                              pools[(state, district)]))
        i += 1

    df = pd.DataFrame(rows)

    # ── a clean pool of indices, handed out without overlap ──
    pool = list(df.index)
    RNG.shuffle(pool)
    cursor = 0

    def take(k):
        nonlocal cursor
        chunk = pool[cursor:cursor + k]
        cursor += k
        return chunk

    def mark(idx, tag):
        """Append a label. Uses a list, so a work can carry two labels
        without the string-mangling bug in v1."""
        df.at[idx, "fraud_label"].append(tag)

    # ─────────────────────────────────────────────────────────
    # 1. COST_OVERRUN — revised far above sanctioned, no scope change
    # ─────────────────────────────────────────────────────────
    for idx in take(C.PLANT["COST_OVERRUN"]):
        mult = RNG.uniform(1.35, 2.8)
        rev = round(df.at[idx, "sanctioned_amount"] * mult, -3)
        df.at[idx, "revised_amount"] = rev
        df.at[idx, "expenditure"] = round(rev * RNG.uniform(0.90, 1.0), -3)
        mark(idx, "COST_OVERRUN")

    # ─────────────────────────────────────────────────────────
    # 2. UNIT_COST_OUTLIER — same work, wildly inflated rate
    # ─────────────────────────────────────────────────────────
    for idx in take(C.PLANT["UNIT_COST_OUTLIER"]):
        amt = round(df.at[idx, "sanctioned_amount"] * RNG.uniform(2.2, 4.2), -3)
        df.at[idx, "sanctioned_amount"] = amt
        df.at[idx, "revised_amount"] = amt
        df.at[idx, "recommended_amount"] = round(amt * RNG.uniform(0.95, 1.02), -3)
        df.at[idx, "expenditure"] = round(amt * RNG.uniform(0.9, 1.0), -3)
        mark(idx, "UNIT_COST_OUTLIER")

    # ─────────────────────────────────────────────────────────
    # 3. GHOST_WORK — complete + fully paid, impossibly fast
    #
    # FIXED from v1: v1 moved completion relative to sanction but left
    # work_start alone, so 44/45 ghosts had completion BEFORE start —
    # an impossible record real portals reject. The detector was
    # catching a data-integrity bug, not fraud.
    # Now the record is internally consistent and the only thing wrong
    # with it is that a toilet block was built in six days.
    # ─────────────────────────────────────────────────────────
    for idx in take(C.PLANT["GHOST_WORK"]):
        san = df.at[idx, "sanction_date"]
        start = san + timedelta(days=RNG.randint(1, 4))
        comp = start + timedelta(days=RNG.randint(2, 9))
        df.at[idx, "work_start_date"] = start
        df.at[idx, "completion_date"] = comp
        df.at[idx, "status"] = "Completed"
        df.at[idx, "expenditure"] = round(
            df.at[idx, "revised_amount"] * RNG.uniform(0.95, 1.0), -3)
        mark(idx, "GHOST_WORK")

    # ─────────────────────────────────────────────────────────
    # 4. DUPLICATE_WORK — one work, two sanctions
    #
    # The clone keeps the SAME description, because it is the same
    # work billed twice. Genuine sibling works in the same district
    # get different place names. That difference is the whole fix.
    # ─────────────────────────────────────────────────────────
    dupes = []
    for idx in take(C.PLANT["DUPLICATE_WORK"]):
        clone = df.loc[idx].copy()
        clone["work_id"] = f"WS/MP{i:05d}/2023-2026/{i:06d}"
        i += 1

        gap = RNG.randint(3, 45)
        clone["sanction_date"] = df.at[idx, "sanction_date"] + timedelta(days=gap)
        clone["recommendation_date"] = df.at[idx, "recommendation_date"] + timedelta(days=gap)
        clone["work_start_date"] = df.at[idx, "work_start_date"] + timedelta(days=gap)
        if not pd.isna(df.at[idx, "completion_date"]):
            clone["completion_date"] = df.at[idx, "completion_date"] + timedelta(days=gap)

        amt = round(df.at[idx, "sanctioned_amount"] * RNG.uniform(0.97, 1.03), -3)
        clone["sanctioned_amount"] = amt
        clone["revised_amount"] = amt
        clone["expenditure"] = round(amt * RNG.uniform(0.85, 1.0), -3)
        clone["fraud_label"] = ["DUPLICATE_WORK"]

        dupes.append(clone)
        mark(idx, "DUPLICATE_WORK")

    df = pd.concat([df, pd.DataFrame(dupes)], ignore_index=True)

    # rebuild the untouched pool after the concat
    clean = [x for x in df.index if not df.at[x, "fraud_label"]]
    RNG.shuffle(clean)
    c = 0

    def take_clean(k):
        nonlocal c
        chunk = clean[c:c + k]
        c += k
        return chunk

    # ─────────────────────────────────────────────────────────
    # 5. STALLED_WORK — sanctioned long ago, no vendor payment raised
    #
    # RENAMED from IDLE_FUNDS. Under the TSA/Hybrid fund flow (Apr 2025)
    # money is released just-in-time to vendors, so there is no parked
    # cash to be "idle". What this actually detects is a sanction that
    # never converted into work: no payment request was ever raised.
    # ─────────────────────────────────────────────────────────
    for idx in take_clean(C.PLANT["STALLED_WORK"]):
        rec = C.FY_START + timedelta(days=RNG.randint(0, 150))
        df.at[idx, "recommendation_date"] = rec
        df.at[idx, "sanction_date"] = rec + timedelta(days=RNG.randint(10, 44))
        df.at[idx, "expenditure"] = 0
        df.at[idx, "status"] = "In Progress"
        # never started, so both downstream dates are blank
        df.at[idx, "work_start_date"] = pd.NaT
        df.at[idx, "completion_date"] = pd.NaT
        df.at[idx, "fraud_label"] = ["STALLED_WORK"]

    # ─────────────────────────────────────────────────────────
    # 6. ROUND_NUMBER — fabricated-looking amounts
    #
    # FIXED from v1: v1 planted five fixed values [5L, 10L, 20L, 25L, 50L]
    # and the detector tested `% 500000 == 0` — the same five values.
    # It was matching a list, not detecting roundness.
    # Now the amounts are random round numbers across a wide range, so
    # the detector has to actually measure how round a number is.
    # ─────────────────────────────────────────────────────────
    for idx in take_clean(C.PLANT["ROUND_NUMBER"]):
        step = RNG.choice([100_000, 500_000, 1_000_000])
        mult = RNG.randint(3, 40)
        amt = float(step * mult)
        df.at[idx, "sanctioned_amount"] = amt
        df.at[idx, "revised_amount"] = amt
        df.at[idx, "recommended_amount"] = amt
        if df.at[idx, "status"] == "Completed":
            df.at[idx, "expenditure"] = round(amt * RNG.uniform(0.9, 1.0), -3)
        df.at[idx, "fraud_label"] = ["ROUND_NUMBER"]

    # ─────────────────────────────────────────────────────────
    # 7. AGENCY_CAPTURE — one agency dominates one district
    #
    # REBALANCED from v1: v1 reassigned 72% of a district and labelled
    # every work of the dominant agency, creating 164 frauds = 29% of
    # all planted fraud from one line of code. The detector found them
    # trivially because it was the same operation inverted.
    # Now: 55% share, and ONLY the works we actually reassigned are
    # labelled — works that already used that agency are left CLEAN.
    # ─────────────────────────────────────────────────────────
    capture_district = "Gaya"
    dominant = "Panchayat Samiti Cell"
    in_district = [x for x in df.index if df.at[x, "district"] == capture_district]
    RNG.shuffle(in_district)

    reassign = [x for x in in_district
                if df.at[x, "implementing_agency"] != dominant]
    target = int(len(in_district) * C.CAPTURE_SHARE)
    already = len(in_district) - len(reassign)
    to_move = max(target - already, 0)

    for idx in reassign[:to_move]:
        df.at[idx, "implementing_agency"] = dominant
        if not df.at[idx, "fraud_label"]:
            df.at[idx, "fraud_label"] = ["AGENCY_CAPTURE"]

    # ── finalise ─────────────────────────────────────────────
    df["fraud_label"] = df["fraud_label"].apply(
        lambda L: "|".join(L) if L else "CLEAN")

    return df.sample(frac=1, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    df.to_csv(C.SYNTHETIC_CSV, index=False)

    print(f"{len(df):,} works generated\n")
    print("FRAUD COMPOSITION")
    print("-" * 46)
    vc = df.fraud_label.value_counts()
    bad = int((df.fraud_label != "CLEAN").sum())
    for k, v in vc.items():
        if k == "CLEAN":
            continue
        print(f"  {k:<26} {v:>4}   {v/bad:>5.1%}")
    print(f"  {'TOTAL PLANTED':<26} {bad:>4}   ({bad/len(df):.1%} of all works)")

    covered = (df.work_description != "").mean()
    print(f"\n  work_description present on {covered:.0%} of works "
          f"(field is optional in eSAKSHI)")