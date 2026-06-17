"""Standalone runner: compare Confirmed Uptrend (bull_regime) WITH vs WITHOUT
the SATA idle-cash dividend variant, for BTC, MSTR and MSTU.

Extracts the backtest functions from app/btc_hourly_app.py via AST (avoiding the
Streamlit UI / network at module import) and runs them against the versioned
data/backtest/ CSVs exactly as the app does.
"""
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import DAILY_MODEL_CT  # noqa: E402

APP = ROOT / "app" / "btc_hourly_app.py"
SRC = APP.read_text()
TREE = ast.parse(SRC)

# ── Streamlit stub: cache_data / cache_resource as passthrough decorators ──
class _St:
    def cache_data(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def deco(fn):
            return fn
        return deco
    cache_resource = cache_data

NS = {
    "pd": pd, "np": np, "joblib": joblib, "json": json, "os": os,
    "st": _St(),
    "yf": None, "requests": None,            # only used in untaken fallbacks
    "_BACKTEST_DATA_DIR": ROOT / "data" / "backtest",
    "DAILY_MODEL_CT": DAILY_MODEL_CT,
    "_BT_LOGIC_VERSION": "sl5-sl5-v29-sata",
    "SATA_ANNUAL_RATE": 0.13,
    "SATA_BUSINESS_DAYS": 250,
    "SATA_DAILY_FACTOR": 1.0 + 0.13 / 250,
    "Path": Path,
}

WANT = [
    "_load_backtest_manifest", "_read_price_csv", "_load_raw_features",
    "_load_btc_prices", "_load_mstr_prices", "_load_mstu_prices",
    "_load_mstu_synthetic", "_build_backtest_preds",
    "run_btc_backtest", "run_mstr_backtest", "run_mstu_backtest",
]
by_name = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
for name in WANT:
    seg = ast.get_source_segment(SRC, by_name[name])
    exec(compile(seg, APP.name, "exec"), NS)

run_btc = NS["run_btc_backtest"]
run_mstr = NS["run_mstr_backtest"]
run_mstu = NS["run_mstu_backtest"]

MODEL_MTIME = float(os.path.getmtime(str(DAILY_MODEL_CT)))
DATA_MTIME = float(os.path.getmtime(str(ROOT / "data" / "backtest" / "manifest.json")))
DATA_END = "2026-06-15"


def stat(bt):
    if bt is None:
        return None
    s = bt["stats"]
    return s


def summarize(label, fn, start, end):
    base = fn(end, start, model_mtime=MODEL_MTIME, data_end=DATA_END,
              data_mtime=DATA_MTIME, entry_gate="bull_regime")
    sata = fn(end, start, model_mtime=MODEL_MTIME, data_end=DATA_END,
              data_mtime=DATA_MTIME, entry_gate="bull_regime_sata")
    sb, ss = stat(base), stat(sata)
    print(f"\n=== {label}  ({start} → {end}) ===")
    if sb is None:
        print("  (no result)")
        return
    print(f"  {'metric':<22}{'No SATA':>16}{'With SATA':>16}")
    rows = [
        ("Final NAV ($)",      sb["final_nav"],   ss["final_nav"]),
        ("Strategy return %",  sb["strat_ret"],   ss["strat_ret"]),
        ("Buy&Hold return %",  sb["bh_ret"],      ss["bh_ret"]),
        ("Alpha vs B&H ($)",   sb["alpha_abs"],   ss["alpha_abs"]),
        ("# trades",           sb["n_trades"],    ss["n_trades"]),
        ("Win rate %",         sb["win_rate"],    ss["win_rate"]),
        ("Max drawdown %",     sb["max_drawdown"],ss["max_drawdown"]),
        ("Sharpe",             sb["sharpe"],      ss["sharpe"]),
        ("Time in market %",   sb["time_in_mkt"], ss["time_in_mkt"]),
        ("After-tax NAV ($)",  sb["after_tax_nav"], ss["after_tax_nav"]),
    ]
    for nm, a, b in rows:
        print(f"  {nm:<22}{a:>16,.2f}{b:>16,.2f}")
    # idle-cash boost
    boost = (ss["final_nav"] / sb["final_nav"] - 1) * 100
    days_cash = (1 - sb["time_in_mkt"] / 100)
    print(f"  → SATA lifts final NAV by {boost:+.2f}%  "
          f"(~{days_cash*100:.0f}% of period in cash)")


if __name__ == "__main__":
    for label, fn, bull_start, bear_start in [
        ("BTC",  run_btc,  "2024-06-01", "2025-06-01"),
        ("MSTR", run_mstr, "2024-06-01", "2025-06-01"),
        ("MSTU", run_mstu, "2024-06-01", "2025-06-04"),
    ]:
        print("\n" + "#" * 70)
        print(f"# {label}")
        print("#" * 70)
        summarize(f"{label} FULL",  fn, bull_start, "2026-05-31")
        summarize(f"{label} BULL",  fn, bull_start, "2025-06-14")
        summarize(f"{label} BEAR",  fn, bear_start, "2026-05-31")
