"""Print the threshold sensitivity table for the real MPLADS dataset."""
import time
import pandas as pd
import adapters as A
import engine as E

t0 = time.time()
raw = pd.read_csv("data/mplads_real.csv", sep=";", dtype=str)
std, meta = A.from_esakshi_export(raw)
df, sc, ev, sk = E.run(std, disable=list(meta["unreliable"]))
elapsed = time.time() - t0

print()
print("REAL MPLADS DATA  —  60,350 works  ·  Rs 3,498 crore")
print("Apr 2023 - Mar 2024  ·  mplads.mospi.gov.in")
print("=" * 58)
print(f"{'THRESHOLD':>10}  {'SURFACED':>9}  {'% OF WORKS':>11}  {'VALUE':>13}")
print("-" * 58)
for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
    f = df.risk_score > t
    val = df.loc[f, "sanctioned_amount"].sum() / 1e7
    print(f"{t:>10.2f}  {int(f.sum()):>9,}  {f.mean():>10.1%}  Rs {val:>8,.0f} cr")
print("=" * 58)
print(f"scored in {elapsed:.0f} seconds")
