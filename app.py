"""
PRAHARI — AI-powered MPLADS Monitoring, Risk & Integrity Platform
Team Outlier · SIH 2026 · Problem Statement SIH26102

Run:  streamlit run app.py
"""

import random
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
import engine as E
import explain as X
import reference as R

st.set_page_config(page_title="PRAHARI — MPLADS Integrity Engine",
                   page_icon="\U0001F6E1", layout="wide")

NAVY, BLUE, RED, AMBER, GREEN, GREY = (
    "#10375C", "#0070C0", "#C0392B", "#E88B00", "#1E8449", "#5A6B7D")
SEV_COLOUR = {"CRITICAL": RED, "HIGH": AMBER, "MEDIUM": BLUE, "LOW": GREY}

st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:11px;
         color:#fff; font-size:0.72rem; font-weight:700; letter-spacing:.3px;}
  .quiet {color:#5A6B7D; font-size:0.85rem;}
</style>""", unsafe_allow_html=True)


def pill(text, colour):
    return f'<span class="pill" style="background:{colour}">{text}</span>'


def lakh(x):
    return f"Rs {x/1e5:,.1f} L"


def crore(x):
    return f"Rs {x/1e7:,.2f} Cr"


# ═════════════════════════════════════════════════ data + engine

@st.cache_data(show_spinner="Scoring the works register\u2026")
def score(csv_path, planted_key, threshold):
    """planted_key only exists to bust the cache when a work is planted."""
    raw = pd.read_csv(csv_path)
    if st.session_state.get("planted"):
        raw = pd.concat([raw, pd.DataFrame(st.session_state.planted)],
                        ignore_index=True)
    df, scores, evidence, skipped = E.run(raw)
    df["flagged"] = df.risk_score > threshold
    return df, scores, evidence, skipped


if "planted" not in st.session_state:
    st.session_state.planted = []
if "picked" not in st.session_state:
    st.session_state.picked = None

# ═════════════════════════════════════════════════ sidebar

with st.sidebar:
    st.markdown(f"### \U0001F6E1 PRAHARI")
    st.caption("MPLADS Monitoring, Risk & Integrity Platform  \n"
               "Team **Outlier** · SIH 2026 · SIH26102")
    st.divider()

    role = st.selectbox("Signed in as", [
        "Ministry (Central Nodal Agency)",
        "State Nodal Authority",
        "District Authority",
        "Member of Parliament"])

    threshold = st.slider(
        "Risk threshold", 0.20, 0.80, float(C.RISK_THRESHOLD), 0.05,
        help="Lower catches more fraud and costs more officer time. "
             "Higher gives a cleaner list and misses more.")

    st.divider()
    st.caption("A flag is a request for review, never an accusation. "
               "Every finding names the document that would clear it.")

df, scores, evidence, skipped = score(
    C.SYNTHETIC_CSV, len(st.session_state.planted), threshold)

# ── role scoping ─────────────────────────────────────────────
scope = df
scope_label = "All India"
if role == "State Nodal Authority":
    state = st.sidebar.selectbox("State", sorted(df.state.unique()))
    scope = df[df.state == state]
    scope_label = state
elif role == "District Authority":
    dist = st.sidebar.selectbox("District", sorted(df.district.unique()))
    scope = df[df.district == dist]
    scope_label = dist
elif role == "Member of Parliament":
    mp = st.sidebar.selectbox("Constituency", sorted(df.constituency.unique()))
    scope = df[df.constituency == mp]
    scope_label = mp

flagged = scope[scope.flagged].sort_values("risk_score", ascending=False)

# ═════════════════════════════════════════════════ header

st.title("PRAHARI \u2014 MPLADS Integrity Engine")
st.caption(f"{role}  \u00b7  scope: **{scope_label}**  \u00b7  "
           f"threshold {threshold:.2f}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Works in scope", f"{len(scope):,}")
k2.metric("Flagged for review", f"{len(flagged):,}",
          f"{len(flagged)/max(len(scope),1):.1%} of works")
k3.metric("Value at risk", crore(flagged.sanctioned_amount.sum()))
k4.metric("Critical", f"{int((flagged.risk_score>=0.80).sum()):,}")
k5.metric("Checks run", f"{len(scores.columns)}")

if skipped:
    st.info("Detectors that sat out on this data source: " +
            ", ".join(f"**{n}** ({why})" for n, why in skipped))

tabs = st.tabs(["Overview", "Work register", "Work detail",
                "Red team \u2014 live test", "Accuracy"])

# ═════════════════════════════════════════════════ 1. OVERVIEW

with tabs[0]:
    if flagged.empty:
        st.success("Nothing above the threshold in this scope.")
    else:
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Where to send people first")
            d = (flagged.groupby(["state", "district"])
                 .agg(works=("work_id", "count"),
                      value=("sanctioned_amount", "sum"),
                      mean_risk=("risk_score", "mean"))
                 .reset_index())
            fig = px.treemap(
                d, path=["state", "district"], values="value",
                color="mean_risk", color_continuous_scale="OrRd",
                custom_data=["works", "mean_risk"])
            fig.update_traces(hovertemplate=(
                "<b>%{label}</b><br>%{customdata[0]} flagged works<br>"
                "Rs %{value:,.0f} at risk<br>mean risk %{customdata[1]:.2f}"
                "<extra></extra>"))
            fig.update_layout(height=380, margin=dict(t=10, l=0, r=0, b=0),
                              coloraxis_colorbar_title="risk")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Box size = rupees at risk.  Colour = mean risk score. "
                       "Click a state to zoom in.")

        with right:
            st.subheader("What is driving the flags")
            mix = (flagged.primary_detector.value_counts()
                   .rename_axis("detector").reset_index(name="works"))
            mix["detector"] = mix.detector.map(
                lambda x: X.PRETTY.get(x, x))
            fig2 = px.bar(mix, x="works", y="detector", orientation="h",
                          color="works", color_continuous_scale="Blues")
            fig2.update_layout(height=250, showlegend=False,
                               coloraxis_showscale=False,
                               margin=dict(t=10, l=0, r=0, b=0),
                               yaxis_title=None, xaxis_title="flagged works")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Risk bands")
            bands = pd.cut(flagged.risk_score,
                           [threshold, 0.60, 0.80, 1.01],
                           labels=["Medium", "High", "Critical"])
            bc = bands.value_counts().reindex(
                ["Critical", "High", "Medium"]).fillna(0).reset_index()
            bc.columns = ["band", "works"]
            fig3 = px.bar(bc, x="band", y="works", color="band",
                          color_discrete_map={"Critical": RED, "High": AMBER,
                                              "Medium": BLUE})
            fig3.update_layout(height=210, showlegend=False,
                               margin=dict(t=10, l=0, r=0, b=0),
                               xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.subheader("Categories carrying the most risk")
        cat = (flagged.groupby("work_name")
               .agg(works=("work_id", "count"),
                    value=("sanctioned_amount", "sum"))
               .sort_values("value", ascending=False).head(8).reset_index())
        cat["work_name"] = cat.work_name.str.slice(0, 52)
        fig4 = px.bar(cat, x="value", y="work_name", orientation="h",
                      text=cat.value.map(crore))
        fig4.update_traces(marker_color=NAVY, textposition="outside")
        fig4.update_layout(height=300, margin=dict(t=10, l=0, r=90, b=0),
                           yaxis_title=None, xaxis_title="value at risk",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)

# ═════════════════════════════════════════════════ 2. REGISTER

with tabs[1]:
    st.subheader(f"{len(flagged):,} works surfaced for review")

    c1, c2, c3 = st.columns(3)
    dets = sorted({d for t in flagged.triggers.fillna("")
                   for d in t.split("|") if d})
    pick_det = c1.multiselect("Signal fired",
                              [X.PRETTY.get(d, d) for d in dets])
    pick_ag = c2.multiselect("Implementing agency",
                             sorted(flagged.implementing_agency.unique()))
    min_amt = c3.number_input("Minimum sanctioned amount (Rs lakh)",
                              0.0, 500.0, 0.0, 5.0)

    view = flagged.copy()
    if pick_det:
        inv = {v: k for k, v in X.PRETTY.items()}
        keys = [inv.get(p, p) for p in pick_det]
        view = view[view.triggers.fillna("").apply(
            lambda t: any(k in t.split("|") for k in keys))]
    if pick_ag:
        view = view[view.implementing_agency.isin(pick_ag)]
    view = view[view.sanctioned_amount >= min_amt * 1e5]

    show = view[["work_id", "district", "ward_or_village", "work_name",
                 "implementing_agency", "sanctioned_amount",
                 "risk_score", "primary_detector", "triggers"]].copy()
    show["primary_detector"] = show.primary_detector.map(
        lambda x: X.PRETTY.get(x, x))
    show = show.rename(columns={
        "work_id": "Work ID", "district": "District",
        "ward_or_village": "Ward / Village", "work_name": "Work",
        "implementing_agency": "Agency",
        "sanctioned_amount": "Sanctioned (Rs)",
        "risk_score": "Risk", "primary_detector": "Primary signal",
        "triggers": "All signals"})

    st.dataframe(
        show, use_container_width=True, hide_index=True, height=430,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="%.2f"),
            "Sanctioned (Rs)": st.column_config.NumberColumn(format="%d")})

    st.download_button(
        "Download this list as CSV",
        view.to_csv(index=False).encode(),
        f"prahari_flagged_{scope_label.replace(' ', '_')}.csv", "text/csv")

    if len(view):
        st.divider()
        chosen = st.selectbox("Open a work in the detail tab",
                              view.work_id.tolist())
        if st.button("Open work detail", type="primary"):
            st.session_state.picked = chosen
            st.rerun()

# ═════════════════════════════════════════════════ 3. WORK DETAIL

with tabs[2]:
    pool = flagged.work_id.tolist()
    if not pool:
        st.info("No flagged works in this scope.")
    else:
        default = (pool.index(st.session_state.picked)
                   if st.session_state.picked in pool else 0)
        wid = st.selectbox("Work", pool, index=default, key="detail_pick")
        idx = df.index[df.work_id == wid][0]
        row = df.loc[idx]
        sigs = E.signals_for(df, scores, evidence, idx)
        note = X.build_note(row, sigs)

        head, meta = st.columns([2, 3])
        with head:
            st.markdown(
                f"### {row.work_id}"
                f"<br>{pill(note['severity'], SEV_COLOUR[note['severity']])}"
                f"&nbsp;&nbsp;<span style='font-size:1.6rem;font-weight:700'>"
                f"{row.risk_score:.2f}</span>", unsafe_allow_html=True)
        with meta:
            st.write("")
            st.markdown(
                f"**{row.work_name}**  \n"
                f"{row.district} \u00b7 {row.ward_or_village} \u00b7 "
                f"{row.location_type}  \n"
                f"{row.implementing_agency}  \u00b7  vendor: {row.vendor_name}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sanctioned", lakh(row.sanctioned_amount))
        m2.metric("Revised", lakh(row.revised_amount))
        m3.metric("Paid", lakh(row.expenditure))
        m4.metric("Status", row.status)

        if isinstance(row.work_description, str) and row.work_description:
            st.caption(f"Work description on eSAKSHI:  \u201c{row.work_description}\u201d")
        else:
            st.caption("Work description: *not filled on eSAKSHI "
                       "(the field is optional)*")

        st.divider()
        st.markdown(f"#### {note['headline']}")

        for f in note["findings"]:
            with st.container(border=True):
                a, b = st.columns([1, 5])
                a.markdown(pill(f["severity"], SEV_COLOUR[f["severity"]]) +
                           f"<br><span class='quiet'>{f['score']:.2f}</span>",
                           unsafe_allow_html=True)
                b.markdown(f"**{f['detector']}**  \n{f['text']}")
                with b.expander("Show the raw numbers"):
                    st.json(f["evidence"])

        st.markdown("#### Recommended action")
        for r in note["resolution"]:
            st.markdown(f"- {r}")

        with st.expander("AI-drafted observation (optional LLM pass)"):
            key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(
                st, "secrets") else ""
            if not key:
                st.caption("No API key configured. The system runs fully on "
                           "templates \u2014 this pass only rewrites the same facts "
                           "as prose, and is never the source of a number.")
                st.code(X.as_text(row, note))
            elif st.button("Draft with Gemini"):
                prose = X.llm_note(row, note, key)
                st.write(prose or "Call failed \u2014 template note stands.")

        st.download_button("Download this audit note",
                           X.as_text(row, note).encode(),
                           f"{row.work_id.replace('/', '_')}_note.txt",
                           "text/plain")

# ═════════════════════════════════════════════════ 4. RED TEAM

with tabs[3]:
    st.subheader("Plant a fraud right now and watch the engine catch it")
    st.caption("The engine has never seen the record you are about to create. "
               "Nothing here is pre-computed.")

    c1, c2 = st.columns(2)
    with c1:
        kind = st.selectbox("Fraud pattern", [
            "Ghost work \u2014 complete and paid in days",
            "Duplicate work \u2014 the same work billed twice",
            "Unit-cost outlier \u2014 real work, inflated price",
            "Stalled work \u2014 sanctioned years ago, never started"])
        district = st.selectbox("District", sorted(df.district.unique()))
    with c2:
        code = st.selectbox("Work type (Annexure-VIII)", list(R.WORKS),
                            format_func=lambda k: f"{k} \u2014 {R.WORKS[k][1][:44]}")
        amount = st.number_input("Sanctioned amount (Rs lakh)",
                                 1.0, 500.0, 24.0, 1.0)

    if st.button("Plant it and re-run the engine", type="primary"):
        rng = random.Random()
        parent, activity, annex, unit, rate, (lo, hi) = R.WORKS[code]
        state = df[df.district == district].state.iloc[0]
        const = df[df.district == district].constituency.iloc[0]
        agency = df[df.district == district].implementing_agency.mode()[0]
        amt = float(amount * 1e5)
        san = C.FY_START + timedelta(days=rng.randint(200, 700))

        row = {
            "work_id": f"RED/TEAM/{rng.randint(1000, 9999)}",
            "state": state, "district": district, "constituency": const,
            "mp_name": df[df.district == district].mp_name.iloc[0],
            "pfms_code": code, "work_category": parent,
            "work_name": activity, "annexure_ref": annex,
            "work_description": f"Live demo work at Main Chowk, {district}",
            "unit": unit, "quantity": round(rng.uniform(lo, hi), 2),
            "location_type": "Rural", "ward_or_village": "Demo Village",
            "implementing_agency": agency, "vendor_name": "Demo Works Pvt Ltd",
            "recommendation_date": san - timedelta(days=30),
            "sanction_date": san, "work_start_date": san + timedelta(days=20),
            "completion_date": pd.NaT,
            "recommended_amount": amt, "sanctioned_amount": amt,
            "revised_amount": amt, "expenditure": amt * 0.4,
            "status": "In Progress", "sc_st_component": False,
            "fraud_label": "RED_TEAM",
        }

        if kind.startswith("Ghost"):
            row["work_start_date"] = san + timedelta(days=2)
            row["completion_date"] = san + timedelta(days=7)
            row["status"] = "Completed"
            row["expenditure"] = amt * 0.98
        elif kind.startswith("Duplicate"):
            twin = dict(row)
            twin["work_id"] = row["work_id"] + "-B"
            twin["sanction_date"] = san + timedelta(days=11)
            twin["sanctioned_amount"] = amt * 1.01
            twin["revised_amount"] = amt * 1.01
            st.session_state.planted.append(twin)
        elif kind.startswith("Unit-cost"):
            row["quantity"] = round(lo, 2)
            row["sanctioned_amount"] = row["revised_amount"] = amt * 3.2
        else:
            row["sanction_date"] = C.FY_START + timedelta(days=20)
            row["work_start_date"] = pd.NaT
            row["expenditure"] = 0

        st.session_state.planted.append(row)
        st.cache_data.clear()
        st.rerun()

    if st.session_state.planted:
        st.divider()
        ids = [r["work_id"] for r in st.session_state.planted]
        res = df[df.work_id.isin(ids)]
        caught = int(res.flagged.sum())
        a, b = st.columns([1, 3])
        a.metric("Planted", len(res))
        a.metric("Caught", caught)
        if caught == len(res) and len(res):
            b.success("Every planted work was surfaced. "
                      "The engine had not seen these records.")
        elif len(res):
            b.warning("Not every planted work cleared the threshold. "
                      "Lower the threshold in the sidebar and look again.")
        for i in res.index:
            r = df.loc[i]
            with st.container(border=True):
                st.markdown(f"**{r.work_id}** \u00b7 risk **{r.risk_score:.2f}** \u00b7 "
                            f"{'FLAGGED' if r.flagged else 'not flagged'}")
                for s in E.signals_for(df, scores, evidence, i):
                    if s.fired:
                        st.markdown(f"- *{X.PRETTY.get(s.detector, s.detector)} "
                                    f"({s.score:.2f})* \u2014 {s.headline}")
        if st.button("Clear planted works"):
            st.session_state.planted = []
            st.cache_data.clear()
            st.rerun()

# ═════════════════════════════════════════════════ 5. ACCURACY

with tabs[4]:
    st.subheader("How we know it works")
    st.caption("Real MPLADS data has no answer key \u2014 nobody has labelled which "
               "works were fraudulent. So accuracy is measured on a synthetic "
               "register where we planted the frauds ourselves and recorded them. "
               "The engine never reads that label.")

    truth_all = df.fraud_label != "CLEAN"
    cap_only = df.fraud_label == "AGENCY_CAPTURE"
    wl = df[~cap_only & (df.fraud_label != "RED_TEAM")]
    truth = wl.fraud_label != "CLEAN"
    flag = wl.flagged
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())

    a, b, c, d = st.columns(4)
    a.metric("Recall", f"{tp/max(int(truth.sum()),1):.1%}")
    b.metric("Precision", f"{tp/max(int(flag.sum()),1):.1%}")
    c.metric("False alarms", f"{fp:,}")
    d.metric("Missed", f"{fn:,}")

    st.caption("Tuned for recall on purpose: a missed fraud costs public money "
               "permanently, a false alarm costs an officer twenty minutes.")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Per detector, against planted ground truth**")
        label = {"ghost_work": "GHOST_WORK", "duplicate": "DUPLICATE_WORK",
                 "unit_cost": "UNIT_COST_OUTLIER", "cost_overrun": "COST_OVERRUN",
                 "stalled_work": "STALLED_WORK", "round_number": "ROUND_NUMBER",
                 "agency_capture": "AGENCY_CAPTURE"}
        rows = []
        for det, tag in label.items():
            actual = df.fraud_label.str.contains(tag, na=False)
            fired = scores[det] > C.FIRE_THRESHOLD
            hit = fired & actual
            rows.append({
                "Detector": X.PRETTY.get(det, det),
                "Planted": int(actual.sum()), "Fired": int(fired.sum()),
                "Recall": hit.sum() / max(actual.sum(), 1),
                "Precision": hit.sum() / max(fired.sum(), 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True,
                     column_config={
                         "Recall": st.column_config.ProgressColumn(
                             "Recall", min_value=0.0, max_value=1.0,
                             format="%.1f%%"),
                         "Precision": st.column_config.ProgressColumn(
                             "Precision", min_value=0.0, max_value=1.0,
                             format="%.1f%%")})
        st.caption("Round numbers is deliberately noisy \u2014 it is weighted so it "
                   "can never surface a work alone. On real MPLADS data a quarter "
                   "of all amounts are exact multiples of Rs 5 lakh.")

    with right:
        st.markdown("**Moving the threshold**")
        pts = []
        for t in [x / 100 for x in range(20, 85, 5)]:
            f = df.risk_score > t
            wlf = f[~cap_only & (df.fraud_label != "RED_TEAM")]
            tp2 = int((wlf & truth).sum())
            pts.append({"threshold": t,
                        "Recall": tp2 / max(int(truth.sum()), 1),
                        "Precision": tp2 / max(int(wlf.sum()), 1)})
        pf = pd.DataFrame(pts)
        fig = go.Figure()
        fig.add_scatter(x=pf.threshold, y=pf.Recall, name="Recall",
                        line=dict(color=GREEN, width=3))
        fig.add_scatter(x=pf.threshold, y=pf.Precision, name="Precision",
                        line=dict(color=BLUE, width=3))
        fig.add_vline(x=threshold, line_dash="dash", line_color=RED,
                      annotation_text="current")
        fig.update_layout(height=320, yaxis_tickformat=".0%",
                          margin=dict(t=20, l=0, r=0, b=0),
                          xaxis_title="risk threshold", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("There is no setting that improves both. The threshold is a "
                   "policy decision about inspection capacity, not a technical one.")

    st.divider()
    st.markdown("**District-level concentration is reported separately**")
    st.caption("Agency concentration describes a district, not a work, so it is "
               "not scored as a work-level detector.")
    conc = df[scores.agency_capture > 0]
    if len(conc):
        cc = (conc.groupby("district")
              .agg(works=("work_id", "count"),
                   dominant=("implementing_agency",
                             lambda s: s.mode()[0]))
              .reset_index())
        st.dataframe(cc, hide_index=True, use_container_width=True)
    else:
        st.write("No district above the concentration floor.")

st.divider()
st.caption("Works data: MPLADS portal (mplads.mospi.gov.in), eSAKSHI. "
           "Guidelines: MPLADS Guidelines 2023, Annexure-VIII. "
           "Detection accuracy measured on a synthetic register with planted "
           "ground truth. \u00b7 Team Outlier \u00b7 SIH 2026")
