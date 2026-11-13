"""
Pilot: Do Big 4 audited banks have different loan loss provisioning?
Reads pilot_data.xlsx (sheet 'Data'), prints results, saves results.txt
Run:  python analysis.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_excel("pilot_data.xlsx", sheet_name="Data")
df.columns = ["bank", "year", "loans", "allowance", "npl", "auditor", "big4",
              "allow_loans", "npl_ratio", "coverage"]

# keep only filled rows, drop the example row
df = df[df["bank"].notna()]
df = df[~df["bank"].astype(str).str.upper().str.startswith("EXAMPLE")]
for c in ["loans", "allowance", "npl", "big4"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["loans", "allowance", "npl", "big4"])
df = df[(df["loans"] > 0) & (df["npl"] > 0)]

df["allow_loans"] = df["allowance"] / df["loans"]
df["npl_ratio"] = df["npl"] / df["loans"]
df["coverage"] = df["allowance"] / df["npl"]
df["log_loans"] = np.log(df["loans"])

out = []
def say(t=""):
    print(t); out.append(str(t))

say(f"Banks/rows used: {len(df)}  (Big4: {int((df.big4==1).sum())}, Non-Big4: {int((df.big4==0).sum())})")
say()
say("Average by auditor type:")
say(df.groupby("big4")[["allow_loans", "npl_ratio", "coverage"]].mean().round(4).to_string())
say()

big, non = df[df.big4 == 1], df[df.big4 == 0]
if len(big) >= 2 and len(non) >= 2:
    t, p = stats.ttest_ind(big["coverage"], non["coverage"], equal_var=False)
    say(f"Coverage difference test (Big4 vs Non-Big4): t = {t:.2f}, p = {p:.3f}")
else:
    say("Need at least 2 Big4 and 2 Non-Big4 rows for the t-test. Add more banks.")

say()
if len(df) >= 10:
    m = smf.ols("allow_loans ~ big4 + npl_ratio + log_loans", data=df).fit()
    say("Regression: Allowance/Loans on Big4, NPL ratio, log(Loans)")
    say(m.summary().tables[1].as_text())
else:
    say("Regression skipped: need at least 10 rows (pilot target is 20 banks).")

say()
say("Note: this is a small pilot. Treat results as indicative, not final.")
open("results.txt", "w", encoding="utf-8").write("\n".join(out))
