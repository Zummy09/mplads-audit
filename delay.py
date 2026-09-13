"""
Delay-risk model — the early-warning layer.

Why this one can be supervised when fraud cannot
------------------------------------------------
Fraud has no labels. Nobody publishes which MPLADS works were
fraudulent, so a supervised fraud model is impossible.

Delay is different. For every COMPLETED work you already know whether
it beat the one-year limit in Guidelines para 3.2.13:

    late = (completion_date - sanction_date) > 365 days

That is a real, verifiable label, free, and thousands of them. So we
train on history and score works that are still running.

The leakage rule
----------------
The model may only see what was knowable AT SANCTION TIME. Using
completion_date or expenditure would be training on the answer. Every
feature below is fixed the day the work is sanctioned.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# XGBoost is the better model on tabular data, but it is a large package and
# cloud installs do fail. scikit-learn ships an equivalent gradient-boosting
# implementation, so a missing optional dependency degrades the model rather
# than taking the whole app down.
try:
    from xgboost import XGBClassifier
    BACKEND = "xgboost"
except ImportError:                                        # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGBClassifier = None
    BACKEND = "sklearn"

import config as C
import models as M

# Known the day the work is sanctioned. Nothing here reveals the outcome.
CATEGORICAL = ["work_name", "district", "implementing_agency",
               "location_type"]
NUMERIC = ["log_amount", "quantity", "rate_ratio", "sanction_month"]


def _features(df, categories=None):
    f = pd.DataFrame(index=df.index)

    amt = df.sanctioned_amount.astype(float)
    f["log_amount"] = np.log10(amt.clip(lower=1))
    f["quantity"] = df.quantity.astype(float) if "quantity" in df else 0.0

    rate = amt / df.quantity.replace(0, np.nan) if "quantity" in df else amt
    med = rate.groupby([df.district, df.work_name]).transform("median")
    f["rate_ratio"] = (rate / med.replace(0, np.nan)).fillna(1.0)

    f["sanction_month"] = df.sanction_date.dt.month.fillna(0).astype(int)

    cats = {}
    for col in CATEGORICAL:
        s = df[col].astype("category")
        if categories and col in categories:
            s = s.cat.set_categories(categories[col])
        cats[col] = s.cat.categories
        f[col] = s.cat.codes            # -1 for unseen values
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0), cats


def label_completed(df):
    """The training set: completed works, labelled late or not."""
    done = df[df.status.str.lower().isin(["completed", "complete"])].copy()
    done = done[done.completion_date.notna() & done.sanction_date.notna()]
    days = (done.completion_date - done.sanction_date).dt.days
    done["days_taken"] = days
    done["late"] = (days > C.COMPLETION_LIMIT_DAYS).astype(int)
    return done


def train(df, verbose=True):
    """Returns (model, categories, metrics) or (None, None, reason)."""
    need = ["sanction_date", "completion_date", "status", "sanctioned_amount"]
    if not M.available(df, need):
        return None, None, {"skipped": "needs sanction and completion dates"}

    done = label_completed(df)
    if len(done) < C.DELAY_MIN_TRAIN or done.late.nunique() < 2:
        return None, None, {"skipped": f"only {len(done)} completed works"}

    X, cats = _features(done)
    y = done["late"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=C.SEED, stratify=y)

    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=C.DELAY_TREES,
            max_depth=C.DELAY_DEPTH,
            learning_rate=C.DELAY_LR,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=C.SEED,
            scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=C.DELAY_TREES,
            max_depth=C.DELAY_DEPTH,
            learning_rate=C.DELAY_LR,
            random_state=C.SEED,
        )
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    metrics = {
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "base_late_rate": float(y.mean()),
        "auc": float(roc_auc_score(yte, p)),
        "avg_precision": float(average_precision_score(yte, p)),
        "backend": BACKEND,
    }
    metrics["verdict"] = (
        "no learnable signal" if metrics["auc"] < 0.55 else
        "weak signal" if metrics["auc"] < 0.65 else
        "usable" if metrics["auc"] < 0.75 else "strong")

    if verbose:
        print(f"  trained on {metrics['train_rows']:,} completed works, "
              f"tested on {metrics['test_rows']:,}")
        print(f"  base late rate  {metrics['base_late_rate']:.1%}")
        print(f"  AUC             {metrics['auc']:.3f}   "
              f"({metrics['verdict']})")
        print(f"  avg precision   {metrics['avg_precision']:.3f}")
        print(f"  backend         {metrics['backend']}")
    return model, cats, metrics


def predict_live(df, model, cats):
    """Delay risk for works still running. Returns a 0-1 Series."""
    live = df[~df.status.str.lower().isin(["completed", "complete"])]
    live = live[live.sanction_date.notna()]
    if not len(live):
        return pd.Series(dtype=float)
    X, _ = _features(live, categories=cats)
    return pd.Series(model.predict_proba(X)[:, 1], index=live.index)


def importances(model, X_columns, top=6):
    """Feature importances, when the backend exposes them."""
    if not hasattr(model, "feature_importances_"):
        return pd.Series(dtype=float)
    imp = pd.Series(model.feature_importances_, index=X_columns)
    return imp.sort_values(ascending=False).head(top)


if __name__ == "__main__":
    df = pd.read_csv(C.SYNTHETIC_CSV)
    for c in M.DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    print("=" * 62)
    print("DELAY-RISK MODEL  \u2014  Guidelines \u00a73.2.13, one-year limit")
    print("=" * 62)
    model, cats, m = train(df)

    if model is None:
        print("  ", m["skipped"])
        raise SystemExit

    print()
    if m["auc"] < 0.55:
        print("  READ THIS: an AUC near 0.500 means the model is guessing.")
        print("  It is not a bug \u2014 it is an honest measurement. If delay does")
        print("  not depend on anything knowable at sanction time, no model")
        print("  can predict it. See what drives delay in this dataset below.")
    else:
        X, _ = _features(label_completed(df))
        imp = importances(model, X.columns)
        if len(imp):
            print("  what the model leans on:")
            for k, v in imp.items():
                print(f"    {k:<22} {v:.3f}")

    risk = predict_live(df, model, cats)
    if len(risk):
        print()
        print(f"  live works scored        {len(risk):,}")
        print(f"  predicted to run late    "
              f"{int((risk > 0.5).sum()):,}  (risk > 0.50)")