"""
=============================================================================
  VIX SENTIMENT RESEARCH SUITE
  github.com/preetx77/Vix-Sentiment
=============================================================================
  Modules
  -------
  1. vix_vs_realized   — implied vs realized volatility premium
  2. sentiment_vs_vix  — Fear & Greed Index vs VIX predictive power (R²)
  3. regime_backtest   — contrarian VIX strategy vs buy-and-hold (Sharpe)
  4. vix_term_structure — VIX curve shape (9D / 30D / 3M / 6M / 1Y)

  Data
  ----
  Live mode  : set USE_LIVE_DATA = True  (requires yfinance + internet)
  Demo mode  : set USE_LIVE_DATA = False (synthetic data, runs anywhere)

  Sources
  -------
  CBOE · MDPI JRFM 2025 · Preprints.org Feb 2026 · ScienceDirect Nov 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
USE_LIVE_DATA = False          # flip to True when running locally with internet
START_DATE    = "2010-01-01"
END_DATE      = "2026-04-01"
TICKERS       = {"vix": "^VIX", "spx": "^GSPC",
                 "vix9d": "^VIX9D", "vix3m": "^VIX3M",
                 "vix6m": "^VIX6M", "vix1y": "^VIX1Y"}

STYLE = {
    "bg":       "#0a0914",
    "surface":  "#12102a",
    "surface2": "#1a1830",
    "text":     "#e8e6f5",
    "muted":    "#9b99b8",
    "dim":      "#4a4870",
    "calm":     "#25c491",
    "normal":   "#ef9f27",
    "stress":   "#f07350",
    "panic":    "#e24b4a",
    "purple":   "#a59fe8",
    "purple2":  "#7c78d8",
    "teal":     "#5dcaa5",
}

# ── DATA LAYER ────────────────────────────────────────────────────────────────

def load_live_data():
    """Pull real data from Yahoo Finance."""
    import yfinance as yf
    raw = {}
    for key, ticker in TICKERS.items():
        s = yf.download(ticker, start=START_DATE, end=END_DATE,
                        progress=False, auto_adjust=True)["Close"].squeeze()
        raw[key] = s
    return pd.DataFrame(raw).dropna(how="all")


def load_synthetic_data():
    """
    Generate realistic synthetic VIX + SPX data that mirrors
    actual market behaviour — use this for demos / offline runs.
    Swap out for load_live_data() when you have internet access.
    """
    np.random.seed(42)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n = len(dates)

    # SPX — geometric Brownian motion with realistic vol clusters
    spx_ret = np.random.normal(0.0003, 0.012, n)
    # inject crisis clusters: 2011, 2015, 2018, 2020, 2022, 2025
    crises = [330, 1350, 2200, 2600, 3200, 3960]
    for c in crises:
        width = np.random.randint(20, 80)
        spx_ret[c:c+width] = np.random.normal(-0.002, 0.025, width)
    spx = 2000 * np.exp(np.cumsum(spx_ret))

    # VIX — mean-reverting (Ornstein-Uhlenbeck), correlated with |spx_ret|
    vix = np.zeros(n)
    vix[0] = 18.0
    theta, kappa, sigma_v = 18.0, 0.08, 3.5
    for i in range(1, n):
        shock = -8 * spx_ret[i] + np.random.normal(0, 0.6)
        vix[i] = max(9, vix[i-1] + kappa*(theta - vix[i-1]) + sigma_v*shock)
    # spike crises
    for c in crises:
        peak = np.random.uniform(35, 65)
        width = np.random.randint(15, 50)
        for j in range(width):
            if c+j < n:
                vix[c+j] = max(vix[c+j], peak * np.exp(-0.05*j))

    # Term structure (VIX9D, VIX3M, VIX6M, VIX1Y) around VIX30
    vix9d = vix * np.random.uniform(0.85, 1.15, n)
    vix3m = vix * np.random.uniform(0.95, 1.10, n)
    vix6m = vix * np.random.uniform(1.00, 1.12, n)
    vix1y = vix * np.random.uniform(1.02, 1.15, n)
    # invert structure during crises
    for c in crises:
        w = 30
        vix9d[c:c+w] = vix[c:c+w] * np.random.uniform(1.05, 1.25, w)
        vix3m[c:c+w] = vix[c:c+w] * np.random.uniform(0.95, 1.05, w)

    # Fear & Greed (0–100, inversely related to VIX with noise)
    fg_raw = 100 - (vix - 9) / (80 - 9) * 100 + np.random.normal(0, 8, n)
    fear_greed = np.clip(fg_raw, 0, 100)

    df = pd.DataFrame({
        "vix":        vix,
        "spx":        spx,
        "vix9d":      vix9d,
        "vix3m":      vix3m,
        "vix6m":      vix6m,
        "vix1y":      vix1y,
        "fear_greed": fear_greed,
    }, index=dates)

    return df


def load_data():
    if USE_LIVE_DATA:
        print("  Fetching live data from Yahoo Finance...")
        df = load_live_data()
        # Fear & Greed not on Yahoo — approximate from VIX
        df["fear_greed"] = np.clip(100 - (df["vix"] - 9) / 71 * 100, 0, 100)
    else:
        print("  Using synthetic data (set USE_LIVE_DATA=True for live data)")
        df = load_synthetic_data()
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────

def vix_regime(v):
    if v < 15:  return "<15",  "Calm",   STYLE["calm"]
    if v < 20:  return "15-20","Normal", STYLE["normal"]
    if v < 30:  return "20-30","Stress", STYLE["stress"]
    return ">30", "Panic", STYLE["panic"]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(STYLE["surface"])
    ax.tick_params(colors=STYLE["muted"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(STYLE["dim"])
    ax.xaxis.label.set_color(STYLE["muted"])
    ax.yaxis.label.set_color(STYLE["muted"])
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=11,
                     fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=STYLE["dim"], alpha=0.3, linewidth=0.5, linestyle="--")


def print_section(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


# ── MODULE 1: VIX VS REALIZED VOLATILITY ─────────────────────────────────────

def module_vix_vs_realized(df, ax1, ax2):
    print_section("MODULE 1 — Implied vs Realized Volatility")

    spx_ret = np.log(df["spx"]).diff()

    # Realized vol: 21-day rolling annualized
    df["realized_vol"] = spx_ret.rolling(21).std() * np.sqrt(252) * 100
    df["implied_vol"]  = df["vix"]
    df["vol_premium"]  = df["implied_vol"] - df["realized_vol"]

    clean = df[["implied_vol","realized_vol","vol_premium"]].dropna()

    pct_positive  = (clean["vol_premium"] > 0).mean() * 100
    avg_premium   = clean["vol_premium"].mean()
    median_premium= clean["vol_premium"].median()

    print(f"  Avg implied vol (VIX):      {clean['implied_vol'].mean():.2f}")
    print(f"  Avg realized vol (21d):     {clean['realized_vol'].mean():.2f}")
    print(f"  Avg vol premium:            {avg_premium:.2f} pts")
    print(f"  Median vol premium:         {median_premium:.2f} pts")
    print(f"  Premium positive:           {pct_positive:.1f}% of trading days")
    print(f"  → VIX overprices fear {pct_positive:.0f}% of the time")

    # Plot 1a: implied vs realized over time
    ax1.fill_between(clean.index, clean["implied_vol"],
                     clean["realized_vol"],
                     where=clean["implied_vol"] >= clean["realized_vol"],
                     alpha=0.25, color=STYLE["panic"], label="Vol premium (sellable)")
    ax1.fill_between(clean.index, clean["implied_vol"],
                     clean["realized_vol"],
                     where=clean["implied_vol"] < clean["realized_vol"],
                     alpha=0.25, color=STYLE["calm"], label="Realized > implied")
    ax1.plot(clean.index, clean["implied_vol"],
             color=STYLE["purple"], linewidth=1.2, label="VIX (implied)")
    ax1.plot(clean.index, clean["realized_vol"],
             color=STYLE["teal"], linewidth=1.0, label="Realized vol (21d)")
    style_ax(ax1, "Implied vs realized volatility",
             ylabel="Volatility (%)")
    ax1.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    # Plot 1b: vol premium distribution
    premium = clean["vol_premium"].dropna()
    ax2.hist(premium, bins=60, color=STYLE["purple2"],
             alpha=0.8, edgecolor="none")
    ax2.axvline(0, color=STYLE["panic"], linewidth=1.5, linestyle="--",
                label="Zero line")
    ax2.axvline(avg_premium, color=STYLE["normal"], linewidth=1.5,
                linestyle="-", label=f"Mean: {avg_premium:.1f}")
    style_ax(ax2, "Vol premium distribution (VIX − realized)",
             xlabel="Premium (pp)", ylabel="Frequency")
    ax2.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    return clean


# ── MODULE 2: SENTIMENT VS VIX (R² COMPARISON) ───────────────────────────────

def module_sentiment_vs_vix(df, ax):
    print_section("MODULE 2 — Sentiment vs VIX Predictive Power (R²)")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    spx_ret = np.log(df["spx"]).diff()

    # Forward returns at multiple horizons
    horizons = {"1M (21d)": 21, "3M (63d)": 63, "6M (126d)": 126}
    signals   = {
        "VIX (implied)":       df["vix"],
        "Fear & Greed Index":  df["fear_greed"],
        "VIX 20d MA":          df["vix"].rolling(20).mean(),
        "Realized vol (21d)":  spx_ret.rolling(21).std() * np.sqrt(252) * 100,
    }

    results = {}
    for sig_name, sig in signals.items():
        results[sig_name] = {}
        for hz_name, hz in horizons.items():
            fwd = df["spx"].pct_change(hz).shift(-hz)
            tmp = pd.DataFrame({"x": sig, "y": fwd}).dropna()
            if len(tmp) < 50:
                results[sig_name][hz_name] = 0.0
                continue
            X, y = tmp[["x"]].values, tmp["y"].values
            r2 = max(0, r2_score(y, LinearRegression().fit(X,y).predict(X)))
            results[sig_name][hz_name] = round(r2 * 100, 2)
            print(f"  {sig_name:25s} | {hz_name:10s} | R² = {r2*100:.2f}%")

    r2_df = pd.DataFrame(results).T
    x = np.arange(len(r2_df))
    width = 0.25
    colors = [STYLE["calm"], STYLE["normal"], STYLE["panic"]]

    for i, (hz_name, col) in enumerate(zip(horizons.keys(), colors)):
        bars = ax.bar(x + i*width, r2_df[hz_name],
                      width=width, color=col, alpha=0.85,
                      label=hz_name, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0.3:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=7, color=STYLE["muted"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(r2_df.index, rotation=12, ha="right", fontsize=9)
    style_ax(ax, "Predictive power: R² by signal & horizon",
             ylabel="R² (%)")
    ax.legend(fontsize=8, facecolor=STYLE["surface2"],
              labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    return r2_df


# ── MODULE 3: REGIME BACKTEST ─────────────────────────────────────────────────

def module_regime_backtest(df, ax1, ax2):
    print_section("MODULE 3 — Contrarian VIX Regime Backtest")

    bt = df[["vix","spx"]].dropna().copy()
    bt["ret"]    = bt["spx"].pct_change()
    bt["regime"] = bt["vix"].apply(lambda v: vix_regime(v)[0])

    # Strategy: go long SPX only when VIX > 30 (contrarian), else flat
    bt["signal"]    = (bt["vix"] > 30).astype(int)
    bt["strat_ret"] = bt["signal"].shift(1).fillna(0) * bt["ret"]
    bt["bh_ret"]    = bt["ret"]

    # Cumulative returns
    bt["strat_cum"] = (1 + bt["strat_ret"]).cumprod()
    bt["bh_cum"]    = (1 + bt["bh_ret"]).cumprod()

    # Stats
    trading_days = 252
    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(trading_days) if r.std() > 0 else 0
    def max_dd(cum):
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        return dd.min() * 100
    def cagr(cum):
        yrs = len(cum) / trading_days
        return (cum.iloc[-1] ** (1/yrs) - 1) * 100 if yrs > 0 else 0

    stats = {
        "Contrarian (VIX>30)": {
            "CAGR %":        round(cagr(bt["strat_cum"]), 2),
            "Sharpe":        round(sharpe(bt["strat_ret"]), 3),
            "Max Drawdown %":round(max_dd(bt["strat_cum"]), 2),
            "Win Rate %":    round((bt["strat_ret"] > 0).mean() * 100, 2),
            "Days Active %": round(bt["signal"].mean() * 100, 2),
        },
        "Buy & Hold": {
            "CAGR %":        round(cagr(bt["bh_cum"]), 2),
            "Sharpe":        round(sharpe(bt["bh_ret"]), 3),
            "Max Drawdown %":round(max_dd(bt["bh_cum"]), 2),
            "Win Rate %":    round((bt["bh_ret"] > 0).mean() * 100, 2),
            "Days Active %": 100.0,
        }
    }

    for strat, s in stats.items():
        print(f"\n  {strat}")
        for k, v in s.items():
            print(f"    {k:20s}: {v}")

    # Forward returns by regime
    bt["fwd_3m"] = bt["spx"].pct_change(63).shift(-63) * 100
    regime_returns = bt.groupby("regime")["fwd_3m"].agg(["mean","median","count"])
    print(f"\n  3-Month Forward Returns by VIX Regime:")
    print(regime_returns.to_string())

    # Plot 3a: cumulative returns
    ax1.plot(bt.index, bt["strat_cum"],
             color=STYLE["purple"], linewidth=1.5, label="Contrarian (VIX>30)")
    ax1.plot(bt.index, bt["bh_cum"],
             color=STYLE["teal"], linewidth=1.0, alpha=0.7, label="Buy & Hold")
    # shade when strategy is active
    ax1.fill_between(bt.index, 0, bt["strat_cum"],
                     where=bt["signal"].astype(bool),
                     alpha=0.08, color=STYLE["purple"])
    style_ax(ax1, "Contrarian VIX strategy vs buy & hold",
             ylabel="Portfolio value (base 1)")
    ax1.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    # Plot 3b: avg 3M returns by regime
    order   = ["<15","15-20","20-30",">30"]
    colors  = [STYLE["calm"], STYLE["normal"], STYLE["stress"], STYLE["panic"]]
    means   = [regime_returns.loc[r,"mean"] if r in regime_returns.index else 0
               for r in order]
    medians = [regime_returns.loc[r,"median"] if r in regime_returns.index else 0
               for r in order]

    x = np.arange(len(order))
    ax2.bar(x - 0.2, means,   width=0.35, color=colors, alpha=0.85, label="Mean")
    ax2.bar(x + 0.2, medians, width=0.35, color=colors, alpha=0.45, label="Median",
            edgecolor=colors, linewidth=1.2)
    ax2.axhline(0, color=STYLE["dim"], linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Calm\n<15","Normal\n15-20",
                          "Stress\n20-30","Panic\n>30"], fontsize=9)
    style_ax(ax2, "Avg 3M S&P 500 forward return by VIX regime",
             ylabel="Return (%)")
    ax2.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    return bt, stats, regime_returns


# ── MODULE 4: VIX TERM STRUCTURE ─────────────────────────────────────────────

def module_vix_term_structure(df, ax1, ax2):
    print_section("MODULE 4 — VIX Term Structure Analysis")

    tenors = ["vix9d","vix","vix3m","vix6m","vix1y"]
    tenor_labels = ["9D","30D","3M","6M","1Y"]

    ts = df[tenors].dropna()

    # Contango ratio: VIX1Y / VIX9D (>1 = normal contango, <1 = backwardation/panic)
    ts["contango_ratio"] = ts["vix1y"] / ts["vix9d"]
    ts["shape"] = ts["contango_ratio"].apply(
        lambda r: "Contango" if r > 1.02 else ("Backwardation" if r < 0.98 else "Flat"))

    contango_pct = (ts["shape"] == "Contango").mean() * 100
    back_pct     = (ts["shape"] == "Backwardation").mean() * 100
    flat_pct     = (ts["shape"] == "Flat").mean() * 100

    print(f"  Term structure shape (% of time):")
    print(f"    Contango (normal):      {contango_pct:.1f}%")
    print(f"    Backwardation (panic):  {back_pct:.1f}%")
    print(f"    Flat:                   {flat_pct:.1f}%")
    print(f"\n  Average term structure:")
    for t, lbl in zip(tenors, tenor_labels):
        print(f"    VIX {lbl:3s}: {ts[t].mean():.2f}  (σ={ts[t].std():.2f})")

    # Plot 4a: term structure curves for 3 market regimes
    calm_mask  = ts["vix"]  < 15
    stress_mask= (ts["vix"] >= 20) & (ts["vix"] < 30)
    panic_mask = ts["vix"]  >= 35

    x = np.arange(len(tenors))
    regime_avgs = {
        "Calm (VIX<15)":     (calm_mask,   STYLE["calm"]),
        "Stress (VIX 20–30)":(stress_mask, STYLE["stress"]),
        "Panic (VIX>35)":    (panic_mask,  STYLE["panic"]),
    }

    for label, (mask, color) in regime_avgs.items():
        if mask.sum() > 10:
            vals = [ts.loc[mask, t].mean() for t in tenors]
            ax1.plot(x, vals, color=color, linewidth=2.0,
                     marker="o", markersize=5, label=label)
            ax1.fill_between(x, vals, alpha=0.08, color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tenor_labels)
    style_ax(ax1, "VIX term structure by market regime",
             xlabel="Tenor", ylabel="VIX level")
    ax1.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    # Plot 4b: contango ratio over time (panic = backwardation events)
    ax2.plot(ts.index, ts["contango_ratio"],
             color=STYLE["muted"], linewidth=0.8, alpha=0.7)
    ax2.axhline(1.0, color=STYLE["dim"], linewidth=1.0, linestyle="--")

    # shade backwardation events
    ax2.fill_between(ts.index, ts["contango_ratio"], 1.0,
                     where=ts["contango_ratio"] < 1.0,
                     color=STYLE["panic"], alpha=0.4, label="Backwardation (panic)")
    ax2.fill_between(ts.index, ts["contango_ratio"], 1.0,
                     where=ts["contango_ratio"] >= 1.0,
                     color=STYLE["calm"], alpha=0.15, label="Contango (normal)")

    style_ax(ax2, "VIX term structure shape: contango ratio (1Y / 9D)",
             ylabel="Ratio")
    ax2.legend(fontsize=8, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])

    return ts


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*62)
    print("  VIX SENTIMENT RESEARCH SUITE")
    print("  github.com/preetx77/Vix-Sentiment")
    print("═"*62)

    # ── Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} trading days  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # ── Build figure: 2×4 grid
    print("\n[2/5] Building charts...")
    plt.rcParams.update({
        "figure.facecolor":  STYLE["bg"],
        "axes.facecolor":    STYLE["surface"],
        "text.color":        STYLE["text"],
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(STYLE["bg"])

    # Header
    fig.text(0.05, 0.97, "VIX SENTIMENT RESEARCH SUITE",
             fontsize=16, fontweight="bold", color=STYLE["text"], va="top")
    fig.text(0.05, 0.945,
             "Implied vol premium · NLP vs VIX predictive power · "
             "Regime backtest · Term structure",
             fontsize=10, color=STYLE["muted"], va="top")
    fig.text(0.95, 0.97,
             "github.com/preetx77/Vix-Sentiment",
             fontsize=9, color=STYLE["dim"], va="top", ha="right")

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.38, wspace=0.35,
                           top=0.92, bottom=0.06,
                           left=0.05, right=0.97)

    ax_m1a = fig.add_subplot(gs[0, 0:2])   # implied vs realized (wide)
    ax_m1b = fig.add_subplot(gs[0, 2])     # premium distribution
    ax_m2  = fig.add_subplot(gs[0, 3])     # R² comparison
    ax_m3a = fig.add_subplot(gs[1, 0:2])   # cumulative returns (wide)
    ax_m3b = fig.add_subplot(gs[1, 2])     # regime forward returns
    ax_m4a = fig.add_subplot(gs[1, 3])     # term structure curves

    # ── Run modules
    print("\n[3/5] Module 1: Implied vs Realized Volatility")
    module_vix_vs_realized(df, ax_m1a, ax_m1b)

    print("\n[4/5] Module 2: Sentiment vs VIX Predictive Power")
    module_sentiment_vs_vix(df, ax_m2)

    print("\n[4/5] Module 3: Regime Backtest")
    module_regime_backtest(df, ax_m3a, ax_m3b)

    print("\n[4/5] Module 4: VIX Term Structure")
    # For term structure, reuse ax_m4a; add a second row using inset
    ax_m4b = ax_m4a.inset_axes([0, -1.25, 1, 0.85])
    module_vix_term_structure(df, ax_m4a, ax_m4b)

    # ── Source strip
    sources = ("Sources: CBOE  ·  MDPI JRFM 2025 (doi:10.3390/jrfm18080412)  ·  "
               "Preprints.org Feb 2026  ·  ScienceDirect Nov 2025  ·  "
               "Data: Yahoo Finance / synthetic demo")
    fig.text(0.5, 0.015, sources, ha="center", fontsize=7.5,
             color=STYLE["dim"])

    print("\n[5/5] Saving output...")
    out_path = "vix_research_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    print(f"\n  Saved → {out_path}")
    print("\n  Done. Open vix_research_output.png to view all charts.\n")
    plt.show()


if __name__ == "__main__":
    main()
