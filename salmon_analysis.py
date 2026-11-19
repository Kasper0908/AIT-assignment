"""
China-Norway 2010 Nobel Prize sanctions: ex-post gravity analysis of salmon trade.

Usage:
    python salmon_analysis.py path/to/BACI_HS02_VXXXXXX.zip   (or the unzipped folder)

Requires: pandas numpy matplotlib pyfixest
    pip install pandas numpy matplotlib pyfixest

Outputs (in ./output):
    salmon_panel.csv          balanced exporter-importer-year panel (zeros filled)
    fig1_descriptive.png      NOR->CHN vs other exporters->CHN, and NOR share of CHN imports
    fig2_event_study.png      year-by-year NOR->CHN effects (ref. year 2010)
    fig2_event_study_ref2009.png   same, ref. year 2009 (robustness)
    table1_main.csv/.txt      main PPML results
    table2_eventstudy*.csv    event-study coefficients
    table3_robustness.csv/.txt  fresh-only and short-window robustness
"""

import sys
import os
import zipfile
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfixest as pf

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SALMON_CODES = ["030212", "030322"]  # HS02: fresh/chilled salmon, frozen Atlantic/Danube salmon. Verify in product_codes file.
NOR, CHN, HKG, VNM = 579, 156, 344, 704
YEAR_MIN, YEAR_MAX = 2002, 2019
SANC_START, SANC_END = 2011, 2016      # sanctions: Oct 2010 prize, normalization Dec 2016
REF_YEAR = 2010                        # event-study reference year
OUT = "output"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load BACI and keep salmon codes only
# ---------------------------------------------------------------------------
def _read_year(fh, frames):
    # k must be read as string: otherwise the leading zero is dropped (030212 -> 30212)
    for chunk in pd.read_csv(fh, dtype={"k": str}, usecols=["t", "i", "j", "k", "v", "q"],
                             chunksize=2_000_000):
        chunk["k"] = chunk["k"].str.strip().str.zfill(6)
        frames.append(chunk[chunk["k"].isin(SALMON_CODES)])


def _year_of(name):
    m = re.search(r"_Y(\d{4})_", os.path.basename(name))
    return int(m.group(1)) if m else None


def load_baci(path):
    """path = the BACI zip file OR the unzipped folder."""
    frames = []
    if os.path.isdir(path):
        files = sorted(os.path.join(root, f) for root, _, fs in os.walk(path)
                       for f in fs if f.endswith(".csv") and _year_of(f))
        for f in files:
            if YEAR_MIN <= _year_of(f) <= YEAR_MAX:
                print(f"Reading {os.path.basename(f)}")
                _read_year(f, frames)
    else:
        with zipfile.ZipFile(path) as z:
            for f in sorted(n for n in z.namelist() if n.endswith(".csv") and _year_of(n)):
                if YEAR_MIN <= _year_of(f) <= YEAR_MAX:
                    print(f"Reading {f}")
                    with z.open(f) as fh:
                        _read_year(fh, frames)
    if not frames:
        sys.exit(f"No BACI year files (BACI_HS02_Yyyyy_...csv) found in {path}")
    df = pd.concat(frames, ignore_index=True)
    df["q"] = pd.to_numeric(df["q"], errors="coerce")  # BACI codes missing quantities as NA
    return df


# ---------------------------------------------------------------------------
# 2. Build balanced panel: all pairs that ever trade salmon, zeros filled
# ---------------------------------------------------------------------------
def build_panel(raw):
    agg = raw.groupby(["t", "i", "j"], as_index=False)[["v", "q"]].sum(min_count=1)
    pairs = agg[["i", "j"]].drop_duplicates()
    years = pd.DataFrame({"t": range(YEAR_MIN, YEAR_MAX + 1)})
    panel = pairs.merge(years, how="cross")
    panel = panel.merge(agg, on=["i", "j", "t"], how="left")
    panel["v"] = panel["v"].fillna(0.0)          # value in 1000 USD
    panel = panel[panel["i"] != panel["j"]]

    panel["pair"] = panel["i"].astype(str) + "_" + panel["j"].astype(str)
    panel["exp_year"] = panel["i"].astype(str) + "_" + panel["t"].astype(str)
    panel["imp_year"] = panel["j"].astype(str) + "_" + panel["t"].astype(str)

    nor_chn = (panel["i"] == NOR) & (panel["j"] == CHN)
    in_window = panel["t"].between(SANC_START, SANC_END)
    panel["sanc"] = (nor_chn & in_window).astype(int)
    panel["post_norm"] = (nor_chn & (panel["t"] > SANC_END)).astype(int)

    # Rerouting extension: NOR -> HKG / VNM during the sanction window
    panel["sanc_hkg"] = ((panel["i"] == NOR) & (panel["j"] == HKG) & in_window).astype(int)
    panel["sanc_vnm"] = ((panel["i"] == NOR) & (panel["j"] == VNM) & in_window).astype(int)

    # Event-study dummies: NOR->CHN x year, omitting REF_YEAR
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        if y != REF_YEAR:
            panel[f"es_{y}"] = (nor_chn & (panel["t"] == y)).astype(int)
    return panel


# ---------------------------------------------------------------------------
# 3. Descriptive figure
# ---------------------------------------------------------------------------
def fig_descriptive(panel):
    to_chn = panel[panel["j"] == CHN]
    nor = to_chn[to_chn["i"] == NOR].groupby("t")["v"].sum() / 1000      # million USD
    other = to_chn[to_chn["i"] != NOR].groupby("t")["v"].sum() / 1000
    share = 100 * nor / (nor + other)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(nor.index, nor.values, marker="o", label="Norway")
    ax[0].plot(other.index, other.values, marker="s", label="All other exporters")
    ax[0].set_title("Salmon imports into China (million USD)")
    ax[1].plot(share.index, share.values, marker="o", color="C3")
    ax[1].set_title("Norway's share of China's salmon imports (%)")
    for a in ax:
        a.axvspan(SANC_START - 0.5, SANC_END + 0.5, color="grey", alpha=0.15, label="Sanction period")
        a.set_xlabel("Year")
        a.set_xticks(range(YEAR_MIN, YEAR_MAX + 1, 2))
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_descriptive.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Regressions (PPML, exporter-year + importer-year + pair FE, clustered by pair)
# ---------------------------------------------------------------------------
FE = "exp_year + imp_year + pair"
VCOV = {"CRV1": "pair"}


def pct(b):
    return 100 * (np.exp(b) - 1)


def run_main(panel):
    models = {
        "(1) Sanction": pf.fepois(f"v ~ sanc | {FE}", data=panel, vcov=VCOV),
        "(2) + post-normalization": pf.fepois(f"v ~ sanc + post_norm | {FE}", data=panel, vcov=VCOV),
        "(3) + rerouting HKG/VNM": pf.fepois(f"v ~ sanc + post_norm + sanc_hkg + sanc_vnm | {FE}",
                                             data=panel, vcov=VCOV),
    }
    rows = []
    for name, m in models.items():
        t = m.tidy()
        for var in t.index:
            b, se, p = t.loc[var, "Estimate"], t.loc[var, "Std. Error"], t.loc[var, "Pr(>|t|)"]
            rows.append({"model": name, "variable": var, "coef": b, "se": se, "p": p,
                         "pct_effect": pct(b), "N": m._N})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/table1_main.csv", index=False)
    with open(f"{OUT}/table1_main.txt", "w") as fh:
        fh.write(res.round(3).to_string(index=False))
    print(res.round(3).to_string(index=False))
    return models


def run_event_study(panel, ref=REF_YEAR, tag=""):
    panel = panel.copy()
    nor_chn = (panel["i"] == NOR) & (panel["j"] == CHN)
    for c in [c for c in panel.columns if c.startswith("es_")]:
        del panel[c]
    for y in range(panel["t"].min(), panel["t"].max() + 1):
        if y != ref:
            panel[f"es_{y}"] = (nor_chn & (panel["t"] == y)).astype(int)
    es_vars = [c for c in panel.columns if c.startswith("es_")]
    m = pf.fepois(f"v ~ {' + '.join(es_vars)} | {FE}", data=panel, vcov=VCOV)
    t = m.tidy().reset_index().rename(columns={"Coefficient": "variable"})
    t["year"] = t["variable"].str.replace("es_", "").astype(int)
    t = pd.concat([t, pd.DataFrame({"year": [ref], "Estimate": [0.0], "Std. Error": [0.0]})])
    t = t.sort_values("year")
    t[["year", "Estimate", "Std. Error"]].to_csv(f"{OUT}/table2_eventstudy{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(t["year"], t["Estimate"], yerr=1.96 * t["Std. Error"], fmt="o", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(SANC_START - 0.5, SANC_END + 0.5, color="grey", alpha=0.15)
    ax.set_title(f"NOR->CHN salmon: PPML event study (ref. {ref}, 95% CI)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient (log points)")
    ax.set_xticks(range(int(t["year"].min()), int(t["year"].max()) + 1, 2))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_event_study{tag}.png", dpi=200)
    plt.close(fig)
    return m


def run_robustness(raw):
    specs = {
        "Baseline (both codes, 2002-2019)": (SALMON_CODES, YEAR_MIN),
        "Fresh salmon only (030212)": (["030212"], YEAR_MIN),
        "Short window (2006-2019)": (SALMON_CODES, 2006),
    }
    rows = []
    for name, (codes, ymin) in specs.items():
        sub = raw[raw["k"].isin(codes) & (raw["t"] >= ymin)]
        p = build_panel(sub)
        p = p[p["t"] >= ymin]
        m = pf.fepois(f"v ~ sanc + post_norm | {FE}", data=p, vcov=VCOV)
        t = m.tidy()
        for var in ["sanc", "post_norm"]:
            b, se, pv = t.loc[var, "Estimate"], t.loc[var, "Std. Error"], t.loc[var, "Pr(>|t|)"]
            rows.append({"spec": name, "variable": var, "coef": b, "se": se, "p": pv,
                         "pct_effect": pct(b), "N": m._N})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/table3_robustness.csv", index=False)
    with open(f"{OUT}/table3_robustness.txt", "w") as fh:
        fh.write(res.round(3).to_string(index=False))
    print(res.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*singleton.*")
    warnings.filterwarnings("ignore", message=".*separation.*")
    if len(sys.argv) < 2:
        sys.exit("Usage: python salmon_analysis.py path/to/BACI_HS02_VXXXXXX.zip   (or the unzipped folder)")
    raw = load_baci(sys.argv[1])
    print(f"Salmon rows: {len(raw):,}")
    panel = build_panel(raw)
    panel.drop(columns=[c for c in panel.columns if c.startswith("es_")]).to_csv(
        f"{OUT}/salmon_panel.csv", index=False)
    print(f"Panel: {panel['pair'].nunique():,} pairs x {panel['t'].nunique()} years = {len(panel):,} obs")
    fig_descriptive(panel)
    run_main(panel)
    run_event_study(panel)
    run_event_study(panel, ref=2009, tag="_ref2009")
    print("\nRobustness (Table 3):")
    run_robustness(raw)
    print(f"Done. See ./{OUT}/")
