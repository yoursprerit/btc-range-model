"""GRID rule × Overall walk-forward replay (see GRID_RETUNE_EVAL.md).

Runs the repo's real ``walkforward_gated_replay`` (V2 allocator, all sleeves,
GRID cap 30 %) once per GRID rule and prints return / Sharpe / max-DD per window.
~1 minute per rule per profile.  Needs the data caches (network on first run) and
``yfinance`` for the BTC sleeves.

    python scripts/eval_grid_portfolio.py [Balanced|Growth|Aggressive]
"""
import sys, warnings
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app")]
warnings.filterwarnings("ignore")
import pandas as pd
import overall_core as oc, ticker_config as tc

CUR = tc.CONFIGS["GRID"]
RULES = {
    "old MACD 10/20/9 (-5%)": replace(CUR, v2_price_gate_ma=0),
    "MACD & close>SMA100 (adopted)": replace(CUR, v2_price_gate_ma=100),
    "MACD & close>SMA200": replace(CUR, v2_price_gate_ma=200),
    "SMA200 +-2% band + 12% trail": replace(CUR, strategy_mode="ma_band", ma_window=200,
                                            band_enter=.02, band_exit=.02, fixed_stop=1.0,
                                            v2_price_gate_ma=0, v2_trail_stop=.12),
}
WINDOWS = [("2021->", "2021-01-01"), ("2023->", "2023-01-01"), ("2025->", "2025-01-01"),
           ("2026 YTD", "2026-01-01"), ("last 3m", (pd.Timestamp.today() - pd.Timedelta(days=91)).strftime("%Y-%m-%d"))]


def window(r, a):
    x = oc.curve_metrics((1 + r[r.index >= pd.Timestamp(a)].fillna(0)).cumprod())
    return f"{x['total_ret']:+7.0%} {x['sharpe']:4.2f} {x['mdd']:6.1%}"


if __name__ == "__main__":
    prof = sys.argv[1] if len(sys.argv) > 1 else "Balanced"
    P = oc.RISK_PROFILES[prof]; caps = oc.caps_for(prof)
    print(f"{prof}: " + " | ".join(w for w, _ in WINDOWS))
    for name, cfg in RULES.items():
        tc.CONFIGS["GRID"] = cfg
        res = oc.run_universe("v2")
        wf = oc.walkforward_gated_replay(
            res, caps=caps, mdd_floor=P["mdd_floor"], objective=P["objective"],
            sata_daily=oc.SATA_DAILY, tilt=True, version="v2",
            cluster_mix=oc.cluster_mix_for(prof, "v2"),
            adds_only_cluster=oc.cluster_adds_band_for(prof, "v2"),
            sata_cap=oc.sata_cap_for("v2"))
        print(f"{name:32s} " + " | ".join(window(wf["ret"], a) for _, a in WINDOWS), flush=True)
    tc.CONFIGS["GRID"] = CUR
