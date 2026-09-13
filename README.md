# PRAHARI — AI-powered MPLADS Monitoring, Risk & Integrity Platform

**Team Outlier** · Smart India Hackathon 2026 · Problem Statement **SIH26102**
Ministry of Statistics and Programme Implementation

---

## What it does

Reads the MPLADS works register, scores every work for risk, and hands an
officer a ranked queue — each flag carrying the numbers behind it and the
document that would clear it.

```
data source ─► adapter ─► 8 detectors ────► risk score ─► audit note ─► dashboard
                       └► 3 guideline rules ──────────► compliance bucket
                       └► delay model ───────────────► early warning
```

Three layers, reported separately on purpose:

| Layer | Question it answers | Output |
|---|---|---|
| **Statistical + ML** | does this look wrong? | a 0–1 risk score |
| **Guideline rules** | does this breach a written clause? | a fact, with the clause |
| **Delay model** | which live works will run late? | a forecast |

A statistical outlier is an opinion — a district officer can argue with it.
A breach of §3.2.4 is arithmetic. Blending a certainty into a probability
produces a number that means neither, so we never do.

---

## Measured accuracy

**92–93% recall at 97–99.5% precision**, across four random seeds.

```
seed 42    93.0% recall   97.2% precision
seed 7     91.9%          98.6%
seed 123   93.3%          98.8%
seed 2026  92.5%          99.5%
```

At seed 42: 630 planted frauds, 603 works surfaced, 586 caught, **17 false
alarms**.

### Why accuracy can be measured at all

Real MPLADS data has no answer key. Nobody has published which works were
fraudulent, so precision and recall cannot be computed on it — ever.

So `generator.py` builds a 5,080-work register and plants 630 frauds
deliberately, recording which rows they are. **The engine never reads that
column.** It is used once, at the end, to score the engine's answers.
Seed-fixed, so anyone can reproduce the identical file.

Validation on synthetic. Application on real.

---

## The checks

### Seven statistical detectors

| Detector | Looks for |
|---|---|
| Ghost work | complete and ≥85% paid within 14 days of sanction |
| Duplicate work | near-twin sanction, same place, matching description |
| Unit-cost outlier | rate per unit far from district peers (median + MAD) |
| Cost overrun | revised ≫ sanctioned, no scope change |
| Stalled work | old sanction, no vendor payment ever raised |
| Round numbers | fabrication signal — corroboration only, capped at 0.80 |
| Agency capture | Herfindahl concentration within a district |

### One unsupervised ML model — `ml.py`

**Isolation Forest.** Needs no labels, which matters because fraud has none.
Catches works where no single value is wrong but the combination appears
nowhere else in the register.

Every feature is **peer-relative** — rate against the district-category
median, duration against peers, and so on. Raw amounts would only tell you
"this is a road".

Measured contribution, against ground truth:

```
rules only   92.1% recall   97.3% precision   596 surfaced
with ML      93.0%          97.2%             603 surfaced
```

**+0.9 points of recall at no cost to precision.** It never leads a finding;
it corroborates. A modest result — and it is only statable because the
measurement was built first.

### Three guideline rules — `compliance.py`

| Clause | Rule | Severity |
|---|---|---|
| §3.2.4 | sanction decision beyond 45 days | VIOLATION |
| §3.2.13 | incomplete beyond one year of sanction | VIOLATION |
| §3.2.9 | below the Rs 2.5 lakh sanction floor | NEEDS JUSTIFICATION |

§3.2.9 is deliberately not called a violation. The Guideline says works
"shall **normally**" be Rs 2.5 lakh or more, and permits less for reasons
recorded in the sanction letter. Those letters are not public, so this is a
count of works requiring a recorded justification — not a count of breaches.

### One supervised model — `delay.py`

Predicts which live works will miss the one-year limit in §3.2.13.

This one **can** be supervised, because delay has a real label: for every
completed work you already know whether it beat 365 days.

```
trained on 3,570 completed works, tested on 1,190
base late rate  22.1%
AUC             0.945
```

Features are restricted to what is knowable **on the day of sanction**.
Using the completion date would be training on the answer.

---

## Findings on real data

60,350 works · Rs 3,498 crore · April 2023 – March 2024 · official portal.

| Finding | Works | Value | Source |
|---|---|---|---|
| Past the 45-day sanction limit | 32,992 | Rs 1,920 cr | §3.2.4 |
| Below the Rs 2.5 lakh floor | 19,735 | Rs 291 cr | §3.2.9 |
| Still unsanctioned after a year | 51,699 (85.7%) | — | portal status |
| Completed | 1,494 (2.5%) | — | portal status |

Measured **as of 2024-03-04**, the newest date in the data. A scraped export
is a snapshot, not a live feed; judging it against today's date would report
every pending work as years overdue, which says more about the download date
than about the ministry.

### Detector output on real data

```
threshold   surfaced    % of works    value
   0.40       5,713        9.5%     Rs 780 cr
   0.60       2,714        4.5%     Rs 393 cr
   0.80         912        1.5%     Rs 248 cr
```

That 9.5% was 40.7% until we traced it to **peer-group density**.
Coincidental near-matches grow with the *square* of group size, and real data
has peer groups of up to 900 works. Two fixes:

- duplicate confidence is scaled by `(6 / group_size) ** 0.5`
- matches with no location evidence at all are rejected — a duplicate claim
  an auditor cannot locate is not actionable

Both left synthetic recall unchanged.

---

## Install and run

```bash
pip install -r requirements.txt

python generator.py                      # build the synthetic register
python evaluate.py                       # recall, precision, per detector
python compliance.py                     # guideline rules, synthetic
python delay.py                          # train and score the delay model
streamlit run app.py                     # the dashboard

# with the real works export in data/
python adapters.py data/mplads_real.csv
python compliance.py real
python threshold_table.py
```

---

## Files

| File | Its one job |
|---|---|
| `config.py` | every tunable number, in one place |
| `reference.py` | Annexure-VIII works, geography, delay drivers |
| `models.py` | the data contract; `Signal` and `FlaggedWork` |
| `generator.py` | build the synthetic register, plant known frauds |
| `adapters.py` | map any real source into the standard schema |
| `detectors.py` | the seven statistical detectors, plus the registry |
| `ml.py` | Isolation Forest, unsupervised anomaly detection |
| `compliance.py` | guideline rules as checkable facts |
| `delay.py` | supervised delay-risk prediction |
| `scoring.py` | many scores in, one risk score out |
| `engine.py` | wire it together: data in, flagged works out |
| `explain.py` | evidence in, plain-language audit note out |
| `evaluate.py` | how well is it working |
| `threshold_table.py` | threshold sensitivity on the real dataset |
| `app.py` | the seven-tab dashboard |

Dependencies point one way. `config`, `reference` and `models` import from
nothing; `app.py` imports everything and nothing imports it. Delete the
dashboard and the engine still runs.

---

## Design decisions worth stating

**Source-agnostic engine.** Detectors declare the columns they need and skip
themselves with a printed reason when a source lacks them, rather than
returning silent zeros. On the public export, four sit out and say so.

**Evidence, not a score.** A detector returning `0.98` cannot produce an
audit note. Every detector returns the peer median, the matching work ID, the
day count — so an officer can act, and defend the action under RTI.

**Never average independent evidence.** A confirmed ghost work trips one
detector at 1.0 and six at zero; an average would score it 0.14 and bury it.
The blend is `0.75 × worst signal + 0.25 × corroboration`.

**Corroborating signals cannot fire alone.** Round numbers caps at 0.80,
which after weighting is 0.36 — below the 0.40 threshold. On real MPLADS data
a quarter of all amounts are exact multiples of Rs 5 lakh, so left uncapped
this detector alone would flag thousands of honest works.

**Tuned for recall.** A missed fraud costs public money permanently. A false
alarm costs an officer twenty minutes. The threshold is a single config value
a ministry tunes to its own inspection capacity.

**Benford's Law was implemented, tested and removed.** It flagged every
district as suspect — not fraud, but the test being misapplied to amounts
spanning only ~2 orders of magnitude.

**The delay model was built, measured at AUC 0.503, and the cause fixed.**
The first generator set duration as `random(45, 400)`, independent of
everything, so there was nothing to learn. Duration now has modelled drivers
— work type, agency capacity, monsoon season, scale — and the model reaches
0.945. **That measures whether the model recovers a known mechanism. It does
not prove delay is predictable in real MPLADS data.** Re-validation against
real completion dates is required before the number is trusted.

**The LLM layer is optional.** Template notes cover every check and cannot
invent a number. The Gemini pass only rewrites facts already computed.
Without an API key the platform runs fully air-gapped.

---

## Known limitations

- The public export carries no completion date, expenditure, revised amount
  or quantity, so four detectors run only against internal eSAKSHI fields.
- It carries no work description either, so duplicate matching falls back to
  ward and village with reduced confidence.
- `IDA` in the public export is the District Authority, not the executing
  agency, so concentration analysis is disabled for that source.
- The official portal exposes only aggregate REST endpoints. Itemized works
  sit behind an OTP-gated citizen form, one member at a time — there is no
  bulk export.
- Accuracy is measured on synthetic data because no labelled real dataset
  exists. It has not been validated against confirmed MPLADS fraud cases,
  because none are published.

---

## Data and attribution

Works data sourced from the MPLADS portal (mplads.mospi.gov.in) via
[Vonter/india-mplads-works](https://github.com/Vonter/india-mplads-works),
Open Database License (ODbL).

Guidelines, Annexure-VIII and the Revised Audit Certificate Format are
published by the Ministry of Statistics and Programme Implementation.

# Tests

```bash
pip install pytest
python generator.py      # the suite scores the generated register
python -m pytest
```

55 tests, about 3 seconds.

## What is covered

| File | What it holds |
|---|---|
| `conftest.py` | `work()` builds one clean work; each test changes only the field it is testing |
| `test_detectors.py` | every detector on handcrafted input, plus the cases that must NOT fire |
| `test_pipeline.py` | score blending, compliance rules, generator, contract, end-to-end, regression floor |

## The tests that matter most

**`test_accuracy_has_not_regressed`** — fails if recall drops below 90% or
precision below 95%. The README claims 92–93% and 97–99.5%; this stops the
claim going stale silently.

**`test_engine_never_reads_the_label`** — greps the engine source for
`fraud_label`. The entire accuracy number rests on the engine not seeing the
answer key, so that guarantee is checked rather than trusted.

**`test_corroborating_detector_cannot_surface_a_work_alone`** — derives each
corroborating detector's cap from the threshold and asserts it holds. This
test found a real inconsistency: `agency_capture` could reach 0.55, above the
0.40 threshold, contradicting the README.

**`test_duplicate_ignores_different_places`** — the case that once produced
214 of 221 false positives.

**`test_ghost_works_are_internally_coherent`** — an early generator logged
completion before the start date on 44 of 45 ghost works, so the detector was
catching a data bug rather than fraud.

## Bugs these tests found while being written

1. `agency_capture` could fire alone, contradicting the documented design.
   Fixed by deriving the cap in `scoring.combine` instead of hard-coding it.
2. `r_sanction_sla` skipped itself when `sanction_date` was entirely null —
   which is exactly the case it exists for.
3. The peer fixture gave every work the same description, so the duplicate
   detector flagged the whole group. The detector was right; the fixture was
   wrong.