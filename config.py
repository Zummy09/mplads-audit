"""
Every tunable number in the system. Nothing here does anything.

If you find yourself wanting to change a number while testing,
it belongs in this file.
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────
# CLOCK
# ─────────────────────────────────────────────────────────────
# Fixed "today" so results are reproducible. If this used the real
# current date, idle/stalled scores would drift daily and you could
# never compare two runs.
TODAY = pd.Timestamp("2026-09-05")

FY_START = pd.Timestamp("2023-04-01")   # eSAKSHI went live 1 Apr 2023

# ─────────────────────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────────────────────
SEED = 42
N_BASE_WORKS = 5000

# How many works carry a Work Description. The eSAKSHI field is
# OPTIONAL, so real data has gaps. The detector must cope.
DESCRIPTION_COVERAGE = 0.70

# Guidelines para 3.2.9: minimum sanction normally not less than Rs 2.5 lakh
MIN_SANCTION = 250_000

# Guidelines para 3.2.13: sanction letter time limit generally 1 year
COMPLETION_LIMIT_DAYS = 365

# Guidelines para 3.2.4: sanction/rejection within 45 days of recommendation
SANCTION_SLA_DAYS = 45

# How many of each fraud pattern to plant
PLANT = {
    "GHOST_WORK":         90,
    "DUPLICATE_WORK":     80,    # pairs, so 160 labelled rows
    "UNIT_COST_OUTLIER": 100,
    "COST_OVERRUN":      100,
    "STALLED_WORK":       90,
    "ROUND_NUMBER":       90,
    "AGENCY_CAPTURE":      1,    # districts, not works
}

# Share of a captured district's works reassigned to one agency.
# Kept moderate so the detector has to work for it.
CAPTURE_SHARE = 0.55

# ─────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────
DATA_DIR      = "data"
SYNTHETIC_CSV = f"{DATA_DIR}/mplads_synthetic.csv"
FLAGGED_CSV   = f"{DATA_DIR}/mplads_flagged.csv"

# ─────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────
RISK_THRESHOLD = 0.40    # above this, surface for human review
FIRE_THRESHOLD = 0.25    # a detector is said to have "fired"

# ─────────────────────────────────────────────────────────────
# DETECTOR WEIGHTS  (ordered by how directly the pattern
# proves money was lost)
# ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "ghost_work":     1.00,   # paid for something that does not exist
    "duplicate":      0.95,   # paid twice for one thing
    "unit_cost":      0.85,   # real work, inflated price
    "cost_overrun":   0.75,   # price grew after approval
    "stalled_work":   0.60,   # waste, not theft
    "isolation_forest": 0.50, # unsupervised ML — catches what no rule anticipates
    "agency_capture": 0.55,   # a risk condition, not a finding
    "round_number":   0.45,   # corroboration only
}

# Cannot cross RISK_THRESHOLD alone: 0.8 * 0.45 = 0.36 < 0.40.
# That is deliberate, not accidental.
# agency_capture scores a DISTRICT, not a work, so it must not
# surface an individual work on its own either.
CORROBORATING = {"round_number", "agency_capture"}

MAX_BLEND = 0.75   # weight on the single worst signal
SUM_BLEND = 0.25   # weight on the corroboration bonus

# ─────────────────────────────────────────────────────────────
# PER-DETECTOR SETTINGS
# ─────────────────────────────────────────────────────────────
GHOST_MAX_DAYS = 14
GHOST_MIN_PAID = 0.85

DUP_WINDOW_DAYS  = 60
DUP_AMOUNT_TOL   = 0.05
DUP_MIN_TEXT_SIM = 0.75

# When there is no description to compare, the evidence is weaker, so we
# demand a closer match before believing it: near-identical amount and a
# shorter window. Without this, sources lacking descriptions (the public
# eSAKSHI export) over-fire badly.
DUP_FALLBACK_TOL    = 0.03
DUP_FALLBACK_WINDOW = 45

# Coincidental near-matches grow with the SQUARE of group size. In a peer
# group of 3 a match is evidence; in a group of 900 it is arithmetic.
# So confidence is scaled down as the group gets denser.
#   factor = (DUP_BASELINE_GROUP / n) ** DUP_DENSITY_POWER, capped at 1.0
DUP_BASELINE_GROUP = 6
DUP_DENSITY_POWER  = 0.5

# Matching on amount and date alone, with NO location evidence at all,
# is the weakest path. Disabled by default: a duplicate claim an auditor
# cannot locate is not actionable. On the real export this alone cut the
# flag rate from 14.6% to 9.5%, with no change to synthetic recall.
# Set True to accept those weaker matches.
DUP_ALLOW_NO_LOCATION = False

UNIT_COST_Z_FLOOR   = 3.5
UNIT_COST_Z_RANGE   = 6.0
UNIT_COST_MIN_PEERS = 5

OVERRUN_FLOOR = 1.25
OVERRUN_RANGE = 1.00

IDLE_MIN_AGE_DAYS = 365
IDLE_AGE_RANGE    = 730

# Calibrated against observed district concentration.
# Normal districts sit at HHI 0.17-0.26 (4-6 active agencies).
# Floor is set just above natural variation; HHI 0.46 scores 1.0.
HHI_FLOOR = 0.26
HHI_RANGE = 0.20

# ─────────────────────────────────────────────────────────────
# ISOLATION FOREST
# ─────────────────────────────────────────────────────────────
ISO_TREES         = 200
ISO_MAX_SAMPLES   = 512
# "assume roughly this share of works are anomalous". Not a claim about
# the real fraud rate — it is the knob that sets how readily the model
# calls something odd.
ISO_CONTAMINATION = 0.05
# Weight 0.50 chosen by measurement, not by feel. On planted ground truth it
# lifts recall 92.1% -> 93.7% with precision unchanged at 99.3%. Above 0.60
# precision starts to fall. At 0.50 a maximum anomaly score reaches 0.50,
# above the 0.40 threshold, so it CAN surface a work alone — which is the
# point of a detector meant to catch what no rule anticipates.

# ─────────────────────────────────────────────────────────────
# DELAY-RISK MODEL
# ─────────────────────────────────────────────────────────────
DELAY_MIN_TRAIN = 300     # below this, not worth training
DELAY_TREES     = 300
DELAY_DEPTH     = 4
DELAY_LR        = 0.08