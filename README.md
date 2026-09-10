# PRAHARI — AI-powered MPLADS Monitoring, Risk & Integrity Platform

**Team Outlier** · Smart India Hackathon 2026 · Problem Statement **SIH26102**
Ministry of Statistics and Programme Implementation

---

## What it does

Reads the MPLADS works register, scores every work for risk, and hands an
officer a ranked queue — each flag carrying the numbers behind it and the
document that would clear it.

```
data source ──► adapter ──► 11 checks ──► risk score ──► audit note ──► dashboard
```

**Accuracy, measured on planted ground truth: 92.1% recall at 99.3% precision.**
584 works surfaced from 4,929, of which 4 were false alarms.

---

## Why accuracy can be measured at all

Real MPLADS data has no answer key. Nobody has published a list of which
works were fraudulent, so precision and recall cannot be computed on it.

So `generator.py` builds a 5,080-work register and plants 630 frauds
deliberately, recording which rows they are in a `fraud_label` column.
**The engine never reads that column.** It is used once, at the end, to
score the engine's answers.

Validation on synthetic data. Application on real data.

---

## The 11 checks

| # | Check | Type | What it looks for |
|---|-------|------|-------------------|
| 1 | Ghost work | statistical | complete and fully paid within days of sanction |
| 2 | Duplicate work | statistical | near-twin sanction, same place, matching description |
| 3 | Unit-cost outlier | statistical | rate far from district peers (median + MAD) |
| 4 | Cost overrun | statistical | revised ≫ sanctioned, no scope change |
| 5 | Stalled work | statistical | old sanction, no vendor payment ever raised |
| 6 | Round numbers | corroborating | fabrication signal — never fires alone |
| 7 | Agency capture | corroborating | Herfindahl concentration within a district |
| 8 | Sanction SLA | **rule** | undecided beyond 45 days — Guidelines §3.2.4 |
| 9 | Completion limit | **rule** | unfinished beyond one year — Guidelines §3.2.13 |
| 10 | Minimum sanction | **rule** | below Rs 2.5 lakh — Guidelines §3.2.9 |
| 11 | Isolation Forest | ML | unsupervised multivariate anomaly *(in build)* |

Rules 8–10 are not opinions. They are countable breaches of written
Guidelines, which is why they are reported separately from statistics.

---

## Findings on real data

60,350 works, Rs 3,498 crore, April 2023 – March 2024, from the MPLADS portal.

| Finding | Count | Value | Source |
|---|---|---|---|
| Past the 45-day sanction limit | 32,992 | Rs 1,920 cr | Guidelines §3.2.4 |
| Below the Rs 2.5 lakh floor | 19,735 | Rs 291 cr | Guidelines §3.2.9 |
| Still unsanctioned after a year | 51,699 (85.7%) | — | portal status |
| Completed | 1,494 (2.5%) | — | portal status |

Note on §3.2.9: the Guideline says works "shall normally" be Rs 2.5 lakh or
more, and allows less for reasons recorded in the sanction letter. Those
letters are not public, so this is a count of works needing a recorded
justification — not a count of violations.

---

## Install and run

```bash
pip install -r requirements.txt

python generator.py     # build the synthetic register
python evaluate.py      # measure recall and precision
streamlit run app.py    # the dashboard
```

Optional, if you have the real works export:

```bash
python adapters.py data/mplads_real.csv
```

---

## Files

| File | Its one job |
|---|---|
| `config.py` | every tunable number, in one place |
| `reference.py` | Annexure-VIII works, geography, description templates |
| `models.py` | the data contract; `Signal` and `FlaggedWork` shapes |
| `generator.py` | build the synthetic register, plant known frauds |
| `adapters.py` | map any real source into the standard schema |
| `detectors.py` | the detectors — each returns a score **and** its evidence |
| `scoring.py` | many scores in, one risk score out |
| `engine.py` | wire it together: data in, flagged works out |
| `explain.py` | evidence in, plain-language audit note out |
| `evaluate.py` | how well is it working? |
| `app.py` | the dashboard |

---

## Design decisions worth stating

**Source-agnostic engine.** Detectors declare the columns they need and skip
themselves with a printed reason when a source lacks them, rather than
returning silent zeros. Running on the public export, four detectors sit out
and say so.

**Evidence, not a score.** A detector returning `0.98` cannot produce an audit
note. Every detector returns the peer median, the matching work ID, the day
count — so an officer can act, and defend the action under the RTI Act.

**Tuned for recall.** A missed fraud costs public money permanently. A false
alarm costs an officer twenty minutes. The threshold is a single config value
a ministry can tune to its inspection capacity.

**Corroborating signals cannot fire alone.** Round numbers scores at most 0.80,
which after weighting is 0.36 — below the 0.40 threshold. On real MPLADS data a
quarter of all amounts are exact multiples of Rs 5 lakh, so left uncapped this
detector would flag thousands of honest works.

**Benford's Law was implemented, tested and removed.** It flagged every district
as suspect. That is not fraud, it is the test being misapplied — our amounts
span only ~2 orders of magnitude. We report what we ruled out as well as what
we kept.

**The LLM layer is optional.** Template notes cover every check and can never
invent a number. The Gemini pass rewrites the same facts as prose. Without an
API key the platform runs fully air-gapped.

---

## Known limitations

- The public export carries no completion date, expenditure, revised amount or
  quantity, so four detectors run only against internal eSAKSHI fields.
- It also carries no work description, so duplicate matching falls back to
  ward/village with reduced confidence. Precision on that source is lower than
  the measured figure above, which is why the description field matters.
- `IDA` in the public export is the District Authority, not the executing
  agency, so concentration analysis is disabled for that source.

---

## Data and attribution

Works data sourced from the MPLADS portal (mplads.mospi.gov.in) via
[Vonter/india-mplads-works](https://github.com/Vonter/india-mplads-works),
Open Database License (ODbL).

Guidelines, Annexure-VIII and the Revised Audit Certificate Format are
published by the Ministry of Statistics and Programme Implementation.
