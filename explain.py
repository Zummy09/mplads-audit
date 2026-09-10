"""
The explanation layer.

Turns a work's Signals into something an officer can act on:

    HEADLINE      what is wrong, in one line
    FINDINGS      each signal's own sentence, with its numbers
    TO RESOLVE    the exact document that would clear the flag

Templates are the default: free, instant, deterministic, and they can
never invent a number. An optional LLM pass rewrites the same facts
into flowing prose. The facts come from the evidence dict either way,
so the LLM is never the source of a figure.
"""

import config as C

# ─────────────────────────────────────────────────────────────
# WHAT WOULD CLEAR THE FLAG
#
# Each entry names a real artefact the eSAKSHI portal already holds.
# This is what turns a suspicion list into a work queue.
# ─────────────────────────────────────────────────────────────

RESOLUTION = {
    "ghost_work": (
        "Retrieve the stage-wise asset photographs uploaded against each vendor "
        "payment request, plus the completion certificate. If the photographs show "
        "progressive construction across the payment stages, clear this flag."),

    "duplicate": (
        "Compare the sanction orders and site locations of both works. If they are "
        "at different locations, or one supersedes the other, clear this flag. If "
        "both were paid for the same asset, refer for recovery."),

    "unit_cost": (
        "Obtain the detailed cost estimate and compare it against the State "
        "Schedule of Rates for this work category. If the rate is justified by site "
        "conditions recorded in the estimate, clear this flag."),

    "cost_overrun": (
        "Obtain the revised administrative sanction and the recorded justification "
        "for the increase. If a scope change was approved before the revision, "
        "clear this flag."),

    "stalled_work": (
        "Ask the Implementing Agency why no vendor payment request has been raised. "
        "If the work is abandoned, initiate cancellation so the entitlement returns "
        "to the MP's available limit."),

    "round_number": (
        "No action on this signal alone. Note it only if another finding on the "
        "same work is confirmed."),

    "agency_capture": (
        "Review how works are allotted to Implementing Agencies in this district. "
        "This is a governance observation about the district, not an allegation "
        "against this work."),
}

SEVERITY = {
    "ghost_work": "CRITICAL", "duplicate": "CRITICAL",
    "unit_cost": "HIGH", "cost_overrun": "HIGH",
    "stalled_work": "MEDIUM", "agency_capture": "LOW",
    "round_number": "LOW",
}

PRETTY = {
    "ghost_work": "Ghost work", "duplicate": "Duplicate work",
    "unit_cost": "Unit-cost outlier", "cost_overrun": "Cost overrun",
    "stalled_work": "Stalled work", "round_number": "Round number",
    "agency_capture": "Agency concentration",
}


def band(score):
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= C.RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _lakh(x):
    return f"Rs {x/1e5:,.1f} lakh"


def build_note(row, signals):
    """row = the work's Series. signals = list of models.Signal, worst first."""
    if not signals:
        return {"headline": "No signal fired on this work.",
                "findings": [], "resolution": [], "severity": "LOW"}

    primary = signals[0]

    headline = (
        f"{PRETTY.get(primary.detector, primary.detector)} suspected \u2014 "
        f"{row.work_name[:60]} in {row.district}, "
        f"{_lakh(row.sanctioned_amount)} sanctioned to "
        f"{row.implementing_agency}."
    )

    findings, resolution, seen = [], [], set()
    for s in signals:
        if not s.fired:
            continue
        findings.append({
            "detector": PRETTY.get(s.detector, s.detector),
            "severity": SEVERITY.get(s.detector, "LOW"),
            "score": s.score,
            "text": s.headline,
            "evidence": s.evidence,
        })
        if s.detector not in seen:
            seen.add(s.detector)
            resolution.append(RESOLUTION.get(s.detector, ""))

    return {"headline": headline, "findings": findings,
            "resolution": [r for r in resolution if r],
            "severity": band(float(row.risk_score))}


def as_text(row, note):
    """Flat text version — for export, and as the LLM prompt input."""
    out = [f"WORK {row.work_id}   risk {row.risk_score}   [{note['severity']}]",
           f"{row.district} / {row.ward_or_village} / {row.implementing_agency}",
           f"{row.work_name}",
           "",
           note["headline"], ""]
    for i, f in enumerate(note["findings"], 1):
        out.append(f"{i}. [{f['severity']}] {f['detector']} ({f['score']:.2f})")
        out.append(f"   {f['text']}")
    out.append("")
    out.append("TO RESOLVE:")
    for r in note["resolution"]:
        out.append(f"  \u2022 {r}")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────
# OPTIONAL LLM PASS
#
# Rewrites the SAME facts as prose. Never given the freedom to
# introduce a number: every figure it may use is already in the
# text we hand it.
# ─────────────────────────────────────────────────────────────

PROMPT = """You are drafting an audit observation for a district officer
reviewing works under the Indian MPLAD Scheme.

Rewrite the findings below as one short paragraph, then a second short
paragraph beginning "Recommended action:".

Rules:
- Use ONLY the numbers given. Do not add, round or invent any figure.
- Neutral official register. This is a matter for verification, not an accusation.
- No bullet points. No headings. Under 130 words total.

FINDINGS
--------
{facts}
"""


def llm_note(row, note, api_key, model="gemini-2.0-flash"):
    """Returns prose, or None if the call fails. Never raises."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        r = m.generate_content(PROMPT.format(facts=as_text(row, note)))
        return r.text.strip()
    except Exception:
        return None
