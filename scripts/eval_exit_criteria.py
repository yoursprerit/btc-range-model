"""Evaluate alternative EXIT criteria vs the live strategy across all periods
and assets (BTC, MSTR, MSTU), using the UI's backtest methodology and data.

Entry gate held fixed at the live default ("bull_regime").  Only the exit logic
and stop-loss configuration change between variants.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import exit_criteria_lib as L

# Live per-asset stop configuration (matches app):
#   BTC  → no stop
#   MSTR → fixed -3% close stop + SL5 re-entry
#   MSTU → fixed -7% close stop + SL5 re-entry
LIVE_STOP = {
    "BTC":  dict(stop_kind=None),
    "MSTR": dict(stop_kind="fixed", stop_pct=0.03, sl5_reentry=True),
    "MSTU": dict(stop_kind="fixed", stop_pct=0.07, sl5_reentry=True),
}

# Locked periods (from app live asset pages)
PERIODS = {
    "BTC":  dict(Bear=("2025-06-01", "2026-05-31"), Bull=("2024-06-01", "2025-06-05"), Full=("2024-06-01", "2026-05-31")),
    "MSTR": dict(Bear=("2025-06-01", "2026-05-31"), Bull=("2024-06-01", "2025-06-05"), Full=("2024-06-01", "2026-05-31")),
    "MSTU": dict(Bear=("2025-06-04", "2026-05-31"), Bull=("2024-06-01", "2025-06-05"), Full=("2024-06-01", "2026-05-31")),
}


def run_variant(asset, exit_fn, stop_override=None, sl5=None):
    cfg = dict(LIVE_STOP[asset])
    if stop_override is not None:
        cfg = dict(stop_override)
    if sl5 is not None:
        cfg["sl5_reentry"] = sl5
    out = {}
    for pname, (s, e) in PERIODS[asset].items():
        r = L.run_backtest(asset, s, e, exit_fn, entry_gate="bull_regime", **cfg)
        out[pname] = r
    return out


def fmt_row(label, res, asset):
    cells = []
    for p in ("Bear", "Bull", "Full"):
        r = res[p]
        if r is None:
            cells.append("   n/a   ")
        else:
            cells.append(f"{r['strat_ret']:+8.1f}%")
    return f"  {label:<34s} " + " ".join(cells)


def bh_row(res):
    cells = []
    for p in ("Bear", "Bull", "Full"):
        r = res[p]
        cells.append(f"{r['bh_ret']:+8.1f}%" if r else "   n/a  ")
    return cells


# ── Variant definitions (exit_fn, stop_override) ──────────────────────────
def variants_for(asset):
    live_stop = dict(LIVE_STOP[asset])
    V = []
    # 0. LIVE baseline
    V.append(("LIVE (TF2 + live stop)", L.exit_tf2, live_stop))
    # 1. Pure signal exits (no stop) — isolate exit logic
    V.append(("TF2, no stop", L.exit_tf2, dict(stop_kind=None)))
    V.append(("TF1 (D2|D3 always), live stop", L.exit_tf1, live_stop))
    V.append(("D3-only (patient), live stop", L.exit_d3_only, live_stop))
    V.append(("TF2+D1 bear exit, live stop", L.exit_with_d1, live_stop))
    V.append(("TF2 D2-confirmed x2, live stop", L.make_exit_d2_confirmed(), live_stop))
    # 2. Trailing-stop family (signal exit = TF2)
    for tp in (0.05, 0.07, 0.10, 0.15):
        V.append((f"TF2 + trail -{int(tp*100)}%", L.exit_tf2,
                  dict(stop_kind="trail", trail_pct=tp, sl5_reentry=True)))
    # 3. Fixed-stop sweep (signal exit = TF2)
    for fp in (0.03, 0.05, 0.07, 0.10):
        V.append((f"TF2 + fixed -{int(fp*100)}%", L.exit_tf2,
                  dict(stop_kind="fixed", stop_pct=fp, sl5_reentry=True)))
    # 4. D3-only + trailing stop (let winners run, cap losers with trail)
    for tp in (0.07, 0.10, 0.15):
        V.append((f"D3-only + trail -{int(tp*100)}%", L.exit_d3_only,
                  dict(stop_kind="trail", trail_pct=tp, sl5_reentry=True)))
    # 5. D3-only + fixed stop
    for fp in (0.05, 0.07, 0.10):
        V.append((f"D3-only + fixed -{int(fp*100)}%", L.exit_d3_only,
                  dict(stop_kind="fixed", stop_pct=fp, sl5_reentry=True)))
    return V


def main():
    summary = {}
    for asset in ("BTC", "MSTR", "MSTU"):
        print("\n" + "=" * 78)
        print(f"  {asset}   (entry gate = bull_regime; periods locked as in UI)")
        print("=" * 78)
        baseline_res = None
        bh_printed = False
        rows = []
        for label, exit_fn, stop in variants_for(asset):
            res = run_variant(asset, exit_fn, stop_override=stop)
            if not bh_printed:
                bh = bh_row(res)
                print(f"  {'Buy & Hold':<34s} " + " ".join(bh))
                print(f"  {'':<34s} {'Bear':>9s} {'Bull':>9s} {'Full':>9s}")
                print("  " + "-" * 74)
                bh_printed = True
            if label.startswith("LIVE"):
                baseline_res = res
            print(fmt_row(label, res, asset))
            rows.append((label, res))
        summary[asset] = (baseline_res, rows)

    # ── Highlight: variants that beat LIVE on ALL THREE periods ───────────
    print("\n\n" + "#" * 78)
    print("  VARIANTS THAT IMPROVE ON LIVE IN ALL THREE PERIODS (Bear & Bull & Full)")
    print("#" * 78)
    for asset in ("BTC", "MSTR", "MSTU"):
        base, rows = summary[asset]
        base_ret = {p: base[p]["strat_ret"] for p in ("Bear", "Bull", "Full")}
        print(f"\n  {asset}  — LIVE: Bear {base_ret['Bear']:+.1f}%  Bull {base_ret['Bull']:+.1f}%  Full {base_ret['Full']:+.1f}%")
        any_found = False
        for label, res in rows:
            if label.startswith("LIVE"):
                continue
            if all(res[p] and res[p]["strat_ret"] > base_ret[p] + 1e-6 for p in ("Bear", "Bull", "Full")):
                deltas = {p: res[p]["strat_ret"] - base_ret[p] for p in ("Bear", "Bull", "Full")}
                print(f"    ✓ {label:<32s} Δ Bear {deltas['Bear']:+.1f}pp  Bull {deltas['Bull']:+.1f}pp  Full {deltas['Full']:+.1f}pp")
                any_found = True
        if not any_found:
            print("    (none — no exit variant strictly dominates LIVE across all three periods)")


if __name__ == "__main__":
    main()
