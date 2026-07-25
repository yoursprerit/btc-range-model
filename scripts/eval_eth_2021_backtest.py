"""ETH on the CURRENTLY SHIPPED strategy, backtested 2021 → now.

The shipped ETH sleeve (spot ETH on the BTC CT regime-divergence signal,
Standard-MA gate, −8% stop — `app/btc_ct_engine.py`) reports from 2024-03-05
only because the committed CT feature CSV starts 2023-11.  This script
answers "what would the backtest look like from 2021?" by rebuilding the
same 116-feature matrix back to 2020-06 and running the deployed engine over
it, unchanged:

  * BTC 12:00-UTC OHLCV      — Binance hourly rebucketed (data-api mirror,
                               full history; the committed pull's own recipe)
  * macro closes (incl. ETH) — Yahoo chart API (same `_yf_requests` path)
  * on-chain (11 series)     — blockchain.info charts, timespan=8years,
                               resampled to daily means
  * Coinbase premium         — Coinbase daily candles vs the BTC 12:00-UTC
                               close, same formula as the pull script
  * 2023-11-01 → now         — the COMMITTED `raw_features_daily.csv` is
                               spliced on top verbatim, so every signal the
                               live app has ever traded is reproduced exactly;
                               the re-fetched history only extends the past.

The engine run is `T.build_preds_offline` → `E.compute_sigs_pure` →
`E._run_bt` with the shipped ETH config (Standard-MA gate, −8% stop), plus
the no-stop variant and Buy&Hold for context.  A validation run over the
shipped window (2024-03-05 →) must reproduce the published +40% / −22.9%
before the 2021→ numbers are worth reading.

⚠️ Read the output caveats: the CT artifact is a DEPLOY-MODE model trained on
2019-03 → 2026-02 data, and the thresholds/gate/stop were tuned on the
2024-03 → window — so a 2021→ backtest is almost entirely in-sample for the
model and cannot be read as out-of-sample performance.  Evaluation only.

Run:  python scripts/eval_eth_2021_backtest.py     (network required)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "app"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import btc_ct_engine as E                       # noqa: E402
import backtest_trailing_stop as T              # noqa: E402
import pull_backtest_data as PBD                # noqa: E402
from eval_eth_native_divergence import fetch_long, metrics, fmt  # noqa: E402

EXT_START = "2020-06-01"          # ≈200d of warmup before the 2021-01-01 start
SPLICE = "2023-11-01"             # committed raw_features takes over here
BT_START = "2021-01-01"
CACHE = _REPO / "tmp"


def _fetch_onchain() -> dict:
    out = {}
    for m in PBD._ONCHAIN:
        col = f"oc_{m.replace('-', '_')}"
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{m}",
                             params={"timespan": "8years", "format": "json",
                                     "sampled": "false"}, timeout=60)
            vals = r.json().get("values", [])
            s = pd.Series({pd.Timestamp(v["x"], unit="s"): v["y"] for v in vals},
                          dtype=float)
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            out[col] = s.groupby(s.index.normalize()).mean().sort_index()
            print(f"    {col}: {len(out[col])} daily rows from {out[col].index[0].date()}")
        except Exception as exc:
            print(f"    [WARN] {m}: {exc}")
        time.sleep(0.1)
    return out


def _fetch_cb(end: str) -> pd.Series:
    rows = []
    cur, stop = pd.Timestamp(EXT_START), pd.Timestamp(end)
    while cur <= stop:
        chunk = min(cur + pd.Timedelta(days=299), stop)
        try:
            r = requests.get(PBD._CB_URL, params={
                "granularity": 86400,
                "start": cur.strftime("%Y-%m-%dT00:00:00Z"),
                "end": (chunk + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
        except Exception as exc:
            print(f"    [WARN] Coinbase chunk {cur.date()}: {exc}")
        cur = chunk + pd.Timedelta(days=1)
        time.sleep(0.15)
    cb = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    cb["date"] = pd.to_datetime(cb["ts"], unit="s").dt.normalize()
    cb = cb.drop_duplicates("date").set_index("date").sort_index()
    cb.index = pd.DatetimeIndex(cb.index).tz_localize(None)
    return cb["close"].astype(float)


def build_extended_features() -> pd.DataFrame:
    """The pull script's raw_features construction from EXT_START, with the
    committed CSV spliced on top from SPLICE (production data wins)."""
    fp = CACHE / "eval_eth_2021_raw_features_ext.csv"
    committed = T.load_raw_features()
    if fp.exists():
        ext = pd.read_csv(fp, index_col=0, parse_dates=True)
    else:
        CACHE.mkdir(exist_ok=True)
        btc = fetch_long("BTCUSDT")                       # 12:00-UTC bars, 2019→
        btc = btc.loc[EXT_START:].ffill()
        df = btc[["close", "high", "low", "volume"]].copy()
        df.columns = ["btc_close", "btc_high", "btc_low", "btc_volume"]
        print("  macro …")
        for nm, sym in PBD._MACRO_SYMS.items():
            d = PBD._yf_requests(sym, EXT_START)
            df[f"{nm}_close"] = d["Close"].reindex(df.index).ffill(limit=7)
            print(f"    {sym}: {len(d)} rows")
        print("  on-chain …")
        for col, s in _fetch_onchain().items():
            df[col] = s.reindex(df.index).ffill(limit=7)
        print("  Coinbase premium …")
        cb_close = _fetch_cb(SPLICE).reindex(df.index)
        prem = (cb_close - df["btc_close"]) / df["btc_close"] * 100
        df["cb_premium"] = prem
        df["cb_premium_ma3"] = prem.rolling(3).mean()
        df["cb_premium_z7"] = (prem - prem.rolling(7).mean()) / prem.rolling(7).std()
        ext = df
        ext.to_csv(fp)
    ext = ext.loc[ext.index < pd.Timestamp(SPLICE), committed.columns]
    full = pd.concat([ext, committed]).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


def eth_prices_2021() -> pd.Series:
    """Spot ETH closes on 12:00-UTC bars: committed CSV from 2023-11 (the
    production prices), Binance-mirror history before that."""
    committed = pd.read_csv(T.DATA / "eth_usd_daily.csv", index_col=0,
                            parse_dates=True)["close"].astype(float)
    long = fetch_long("ETHUSDT")["close"].astype(float)
    pre = long.loc[long.index < committed.index[0]]
    return pd.concat([pre, committed]).sort_index()


def main() -> int:
    print("Building extended feature matrix (2020-06 → now, committed CSV "
          "spliced from 2023-11) …", flush=True)
    rf = build_extended_features()
    nn = rf.loc[:SPLICE].notna().mean().mean() * 100
    print(f"  {len(rf)} rows  {rf.index[0].date()} → {rf.index[-1].date()}  "
          f"(pre-splice avg non-null {nn:.0f}%)")

    print("Running the deployed CT engine over the extended history …", flush=True)
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, "2020-10-01", str(rf.index[-1].date()))
    dates = pd.DatetimeIndex(comp["target_date"])
    sigs = E.compute_sigs_pure(comp)
    eth = eth_prices_2021()
    px = eth.reindex(dates).ffill().to_numpy(float)
    print(f"  signal bars: {dates[0].date()} → {dates[-1].date()}  ({len(dates)})")

    def run(stop, start):
        s = dict(sigs); s["tf2_entry"] = sigs["tf2_entry_ma"]     # Standard-MA gate
        bt = E._run_bt(dates, px, s, stop, pd.Timestamp(start))
        nav = bt["nav"][bt["nav"].index >= pd.Timestamp(start)]
        return nav, bt["trades"]

    # ── validation: the shipped window must reproduce the published numbers ──
    nav_v, tr_v = run(0.08, "2024-03-05")
    m_v = metrics(nav_v)
    print("\nVALIDATION — shipped window 2024-03-05 → now on the extended matrix "
          "(published: +40.0% / −22.9% / Sh 0.52):")
    print(f"  CT-MA −8%: {fmt(m_v)}  tr {len([t for t in tr_v if t['exit_date'] >= pd.Timestamp('2024-03-05')])}")

    # ── the answer: 2021 → now ───────────────────────────────────────────────
    nav8, tr8 = run(0.08, BT_START)
    nav0, tr0 = run(None, BT_START)
    bh = eth.loc[BT_START:]

    print("\n" + "═" * 92)
    print(f"ETH 2021-01-01 → {dates[-1].date()} — CURRENT strategy "
          "(BTC CT signal, Standard-MA gate) vs Buy & hold")
    print("═" * 92)
    wr8 = 100 * np.mean([t["ret"] > 0 for t in tr8]) if tr8 else 0.0
    print(f"  {'CT-MA −8% stop (shipped)':28s} {fmt(metrics(nav8))}  "
          f"tr {len(tr8):3d}  wr {wr8:3.0f}%")
    wr0 = 100 * np.mean([t["ret"] > 0 for t in tr0]) if tr0 else 0.0
    print(f"  {'CT-MA no stop':28s} {fmt(metrics(nav0))}  tr {len(tr0):3d}  wr {wr0:3.0f}%")
    print(f"  {'Buy & hold':28s} {fmt(metrics(bh))}")

    print("\nPer-year breakdown (strategy = shipped config; each year rebased):")
    print(f"  {'year':6s} {'CT-MA −8% (shipped)':>44s}   {'Buy & hold':>40s}  trades")
    for y in range(2021, dates[-1].year + 1):
        y0, y1 = f"{y}-01-01", f"{y}-12-31"
        seg = nav8.loc[y0:y1]; bhs = bh.loc[y0:y1]
        ntr = len([t for t in tr8 if pd.Timestamp(y0) <= t["exit_date"] <= pd.Timestamp(y1)])
        if len(seg) < 10:
            continue
        print(f"  {y:<6d} {fmt(metrics(seg))}   {fmt(metrics(bhs))}  {ntr:3d}")

    print("\nRepo-style periods:")
    for lbl, s0 in (("🐻 Bear 2021-22 (2021→2022-12)", None),
                    ("🐂 Bull 2023 → now", "2023-01-01"),
                    ("🔬 Recent 2025 → now", "2025-01-01")):
        if s0 is None:
            seg, bhs = nav8.loc["2021-01-01":"2022-12-31"], bh.loc["2021-01-01":"2022-12-31"]
        else:
            seg, bhs = nav8.loc[s0:], bh.loc[s0:]
        print(f"  {lbl:34s} strat {fmt(metrics(seg))} | B&H {fmt(metrics(bhs))}")

    print("\nTrade log (shipped config, 2021 → 2024-03-05 — the reconstructed era):")
    for t in tr8:
        if t["exit_date"] < pd.Timestamp("2024-03-05"):
            print(f"  {t['entry_date'].date()} → {t['exit_date'].date()}  "
                  f"{t['ret']*100:+7.1f}%  {t['reason']}")

    print("""
CAVEATS — why this is NOT comparable to the app's published OOS numbers:
  * The CT artifact is the deploy-mode model TRAINED on 2019-03 → 2026-02
    data; 2021 → 2026-02 is in-sample for the H/L ensemble.
  * The thresholds (U1 +1.3 / D2 −1.3), Standard-MA gate and −8% stop were
    chosen on the 2024-03 → window — also in-sample for the early years.
  * Pre-2023-11 features are re-fetched today (Yahoo / blockchain.info daily
    means / Coinbase candles / Binance-mirror BTC bars), not a production
    vintage; pre-2023-11 signals are therefore approximate.  From 2023-11 the
    committed CSV is used verbatim and the validation row must match.
  * Pre-2023-11 ETH prices come from the global-Binance mirror (≤0.6% venue
    difference vs the committed binance.us bars).
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
