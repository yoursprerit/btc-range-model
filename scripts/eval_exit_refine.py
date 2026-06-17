"""Finer, risk-aware sweep of the most promising exit variants per asset.
Reports return / max-DD / Sharpe for Bear, Bull, Full and flags strict
all-period improvements over LIVE."""
import warnings
warnings.filterwarnings("ignore")
import exit_criteria_lib as L
from eval_exit_criteria import PERIODS, LIVE_STOP


def run(asset, exit_fn, stop):
    out = {}
    for p, (s, e) in PERIODS[asset].items():
        out[p] = L.run_backtest(asset, s, e, exit_fn, entry_gate="bull_regime", **stop)
    return out


def grid(asset):
    live = dict(LIVE_STOP[asset])
    g = [("LIVE", L.exit_tf2, live)]
    # BTC-focused patient + wide trail/fixed
    for tp in (0.10, 0.12, 0.15, 0.18, 0.20, 0.25):
        g.append((f"D3-only + trail -{int(tp*100)}%", L.exit_d3_only,
                  dict(stop_kind="trail", trail_pct=tp, sl5_reentry=True)))
    for fp in (0.08, 0.10, 0.12, 0.15, 0.20):
        g.append((f"D3-only + fixed -{int(fp*100)}%", L.exit_d3_only,
                  dict(stop_kind="fixed", stop_pct=fp, sl5_reentry=True)))
    # TF2 + fixed fine grid (leveraged sweet-spot search)
    for fp in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08):
        g.append((f"TF2 + fixed -{int(fp*100)}%", L.exit_tf2,
                  dict(stop_kind="fixed", stop_pct=fp, sl5_reentry=True)))
    # TF2 + wide trail
    for tp in (0.12, 0.15, 0.20):
        g.append((f"TF2 + trail -{int(tp*100)}%", L.exit_tf2,
                  dict(stop_kind="trail", trail_pct=tp, sl5_reentry=True)))
    return g


def main():
    for asset in ("BTC", "MSTR", "MSTU"):
        print("\n" + "=" * 92)
        print(f"  {asset}")
        print("=" * 92)
        print(f"  {'variant':<28s} | {'Bear ret/DD/Shp':>22s} | {'Bull ret/DD/Shp':>22s} | {'Full ret/DD/Shp':>22s}")
        print("  " + "-" * 88)
        base = None
        rows = []
        for label, fn, stop in grid(asset):
            r = run(asset, fn, stop)
            rows.append((label, r))
            if label == "LIVE":
                base = r

            def cell(p):
                x = r[p]
                return f"{x['strat_ret']:+7.1f}/{x['max_dd']:5.0f}/{x['sharpe']:4.1f}"
            print(f"  {label:<28s} | {cell('Bear'):>22s} | {cell('Bull'):>22s} | {cell('Full'):>22s}")

        br = {p: base[p]["strat_ret"] for p in ("Bear", "Bull", "Full")}
        print("  " + "-" * 88)
        print(f"  Strict all-period improvements over LIVE (Bear {br['Bear']:+.1f} / Bull {br['Bull']:+.1f} / Full {br['Full']:+.1f}):")
        found = False
        for label, r in rows:
            if label == "LIVE":
                continue
            if all(r[p]["strat_ret"] > br[p] + 1e-6 for p in ("Bear", "Bull", "Full")):
                d = {p: r[p]["strat_ret"] - br[p] for p in ("Bear", "Bull", "Full")}
                print(f"    ✓ {label:<26s} Δ {d['Bear']:+.1f} / {d['Bull']:+.1f} / {d['Full']:+.1f} pp")
                found = True
        if not found:
            print("    (none)")


if __name__ == "__main__":
    main()
