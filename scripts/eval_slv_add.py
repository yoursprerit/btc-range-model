"""Would adding SLV (iShares Silver Trust) as a new app improve the strategy?

Evaluation only — nothing here is wired into ticker_config.py or overall_core.py.
This mirrors the sleeve half of scripts/eval_leveraged_add.py: find SLV's best
strategy with the SAME sweep the other ETF apps use, score its sleeve OOS, and
measure how redundant it is with the book's existing precious-metals sleeve.

Data is read from the committed daily snapshots (SLV OHLC lives in
data/remx/macro_daily.csv and data/gldm/gldm_macro_daily.csv as slv_* columns),
so it runs fully offline. Only the price-driven trend engines (ma / dual_ma /
macd / ma_vol) are swept — the divergence Pure-Regime engine needs the trained
daily High/Low model, which is not built here (see the doc's caveat).

    python scripts/eval_slv_add.py

Findings are written up in SLV_ADDITION_EVAL.md.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import backtest_ticker as B
from app.ticker_config import TickerConfig, _STD_PERIODS

OOS = "2021-01-01"

# ── cached SLV daily OHLC (REMX file has the longest history: 2015→2026) ──
df = pd.read_csv("data/remx/macro_daily.csv", index_col=0, parse_dates=True)
slv = df[["slv_open", "slv_high", "slv_low", "slv_close"]].dropna()
preds = pd.DataFrame({"target_date": pd.to_datetime(slv.index),
                      "px_close": slv["slv_close"].to_numpy(float)}).reset_index(drop=True)
print(f"SLV daily bars: {len(preds)}  "
      f"{preds.target_date.iloc[0].date()} → {preds.target_date.iloc[-1].date()}")


def mk_cfg(mode, **kw):
    base = dict(key="SLV", name="iShares Silver Trust", blurb="", emoji="🥈",
                accent="#9ca3af", accent_dark="#4b5563", accent_bg="#f9fafb",
                accent_bg2="#f3f4f6", primary_symbol="SLV", macro_syms={}, extra_syms={},
                sentiment=[], sentiment_label="", traded_assets=[("SLV", "px_close")],
                asset_labels={"px_close": "SLV"}, strategy_mode=mode, strategy_name=mode,
                ma_window=kw.get("ma_window", 100), fixed_stop=kw.get("stop", 0.05),
                fetch_start="2015-01-01", oos_start=OOS, periods=_STD_PERIODS)
    for k in ("ma_fast", "ma_slow", "macd_fast", "macd_slow", "macd_signal",
              "vol_win", "vol_med_win", "vol_k"):
        if k in kw:
            base[k] = kw[k]
    return TickerConfig(**base)


def run(cfg, spec, oos_start=OOS, end=None):
    if spec["mode"] == "ma":
        r = B.simulate_regime(cfg, preds, None, "px_close", ma_window=spec["ma_window"],
                              stop_pct=spec["stop"], oos_start=oos_start, end=end)
    else:
        r = B.simulate_regime(cfg, preds, None, "px_close", stop_pct=spec["stop"],
                              oos_start=oos_start, end=end)
    m = B._metrics(r["strat"], r["dates"]); bh = B._metrics(r["bh"], r["dates"])
    tr = r["trades"]
    return m, bh, (float((tr > 0).mean() * 100) if len(tr) else 0.0), len(tr)


# ── candidate universe (mirrors + extends sweep_asset) ────────────────────
cands = []
for w in (20, 30, 40, 50, 60, 80, 100, 120, 150, 200):
    for stop in (0.05, 1.0):
        cands.append(dict(mode="ma", ma_window=w, stop=stop, label=f"MA{w} stop={stop}"))
for fast, slow in [(10, 50), (20, 50), (20, 100), (25, 100), (30, 120), (50, 100), (50, 150), (50, 200)]:
    for stop in (0.05, 1.0):
        cands.append(dict(mode="dual_ma", ma_fast=fast, ma_slow=slow, stop=stop,
                          label=f"DualMA {fast}/{slow} stop={stop}"))
for f, s, sg in [(10, 20, 9), (12, 26, 9), (8, 17, 9), (5, 35, 5)]:
    for stop in (0.05, 1.0):
        cands.append(dict(mode="macd", macd_fast=f, macd_slow=s, macd_signal=sg, stop=stop,
                          label=f"MACD {f}/{s}/{sg} stop={stop}"))
for w in (50, 100, 150, 200):
    for vk in (0.9, 0.95, 1.0, 1.1):
        for stop in (0.05, 1.0):
            cands.append(dict(mode="ma_vol", ma_window=w, vol_win=10, vol_med_win=189,
                              vol_k=vk, stop=stop, label=f"MA{w}+vol k={vk} stop={stop}"))

rows = []
for spec in cands:
    kw = {k: v for k, v in spec.items() if k not in ("mode", "label")}
    cfg = mk_cfg(spec["mode"], **kw)
    fm, fbh, _, _ = run(cfg, spec, oos_start="2015-01-01", end=None)
    om, obh, owr, on = run(cfg, spec, oos_start=OOS, end=None)
    beats_loss = mdd_ok = participate = True
    for lbl, s, e in _STD_PERIODS:
        m, bh, _, _ = run(cfg, spec, oos_start=s, end=e)
        if bh["total_ret"] < 0 and m["total_ret"] < bh["total_ret"] - 1e-9:
            beats_loss = False
        if m["mdd"] < bh["mdd"] - 1e-9:
            mdd_ok = False
        if bh["total_ret"] > 0.05 and m["total_ret"] < 0.55 * bh["total_ret"]:
            participate = False
    rows.append(dict(label=spec["label"], mode=spec["mode"],
                     oos_ret=om["total_ret"], oos_mdd=om["mdd"], oos_sharpe=om["sharpe"],
                     oos_wr=owr, oos_n=on, full_ret=fm["total_ret"], full_mdd=fm["mdd"],
                     full_sharpe=fm["sharpe"], bh_oos_ret=obh["total_ret"],
                     bh_oos_mdd=obh["mdd"], bh_oos_sharpe=obh["sharpe"],
                     beats_loss=beats_loss, mdd_ok=mdd_ok, participate=participate))
R = pd.DataFrame(rows)
tiers = [R[R.beats_loss & R.mdd_ok & R.participate], R[R.beats_loss & R.mdd_ok],
         R[R.mdd_ok & R.participate], R[R.mdd_ok], R]
pool = next(t for t in tiers if len(t))
best = pool.sort_values(["oos_ret", "oos_sharpe"], ascending=False).iloc[0]
bh = R.iloc[0]

print(f"\nSLV Buy&Hold OOS 2021→now: ret={bh.bh_oos_ret*100:+.0f}%  "
      f"MDD={bh.bh_oos_mdd*100:.0f}%  Sharpe={bh.bh_oos_sharpe:.2f}")
print("\nTop 8 by OOS return (best-satisfying tier):")
for _, x in pool.sort_values(["oos_ret", "oos_sharpe"], ascending=False).head(8).iterrows():
    print(f"  {x.label:26s} OOS ret={x.oos_ret*100:+7.0f}% MDD={x.oos_mdd*100:6.0f}% "
          f"Sh={x.oos_sharpe:.2f} WR={x.oos_wr:3.0f}% n={x.oos_n:2d}")
print("\nTop 6 by OOS Sharpe (robustness lens):")
for _, x in R.sort_values("oos_sharpe", ascending=False).head(6).iterrows():
    print(f"  {x.label:26s} OOS ret={x.oos_ret*100:+7.0f}% MDD={x.oos_mdd*100:6.0f}% "
          f"Sh={x.oos_sharpe:.2f} WR={x.oos_wr:3.0f}% n={x.oos_n:2d}")
print(f"\n>>> best return sleeve : {best.label}  "
      f"({best.oos_ret*100:+.0f}% / MDD {best.oos_mdd*100:.0f}% / Sharpe {best.oos_sharpe:.2f})")

# ── redundancy: SLV vs the book's existing precious-metals sleeve ──────────
g = pd.read_csv("data/gldm/gldm_macro_daily.csv", index_col=0, parse_dates=True)
cols = {"GLDM (gold)": "gldm_close", "GDX (gold miners)": "gdx_close",
        "UGL (2x gold)": "ugl_close", "Gold futures": "gc_close", "SLV": "slv_close"}
G = pd.DataFrame({k: g[v] for k, v in cols.items()}).dropna()
Rg = np.log(G[G.index >= OOS]).diff().dropna()
print("\nSLV daily-return correlation OOS 2021→now (redundancy check):")
for c in G.columns:
    if c != "SLV":
        print(f"  SLV ↔ {c:20s} {Rg['SLV'].corr(Rg[c]):+.2f}")
