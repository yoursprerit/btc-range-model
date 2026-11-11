"""GRID strategy retune study (2026-11) — see GRID_RETUNE_EVAL.md.

Self-contained (pandas/numpy only) long/flat backtester on data/grid/macro_daily.csv
using the repo's fill convention (decide at close i-1, fill at close i), run ONCE
over the whole history and sliced into the reporting windows, with a 10 bp/side fee.
Compares the live MACD 10/20/9 (−5% stop) with MACD re-tunes, a crash-shield, the
banded-SMA family and a banded-SMA|MACD hybrid, plus a pre-2021 walk-forward pick.

    python scripts/eval_grid_retune.py
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_csv(ROOT / "data/grid/macro_daily.csv", index_col=0, parse_dates=True)
C = D.px_close.to_numpy(float); DATES = D.index.to_numpy(); FEE = 0.001
WINDOWS = {"full 2015→": "2015-01-01", "OOS 2021→": "2021-01-01",
           "bull 2023→": "2023-01-01", "recent 2025→": "2025-01-01"}


def rm(x, w): return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()
def ema(x, s): return pd.Series(x).ewm(span=s, adjust=False).mean().to_numpy()


def macd_sig(f, s, g):
    m = ema(C, f) - ema(C, s)
    return (m - ema(m, g)) > 0


def ma_band(n, be, bx):
    sm = rm(C, n); on = False; out = np.zeros(len(C), bool)
    for i in range(len(C)):
        if (not on) and C[i] > sm[i] * (1 + be): on = True
        elif on and C[i] < sm[i] * (1 - bx): on = False
        out[i] = on
    return out


def crash(x, m, hw=252):
    hi = pd.Series(C).rolling(hw, min_periods=60).max().to_numpy(); sm = rm(C, m)
    on = True; out = np.ones(len(C), bool)
    for i in range(len(C)):
        if on and C[i] < (1 - x) * hi[i]: on = False
        elif (not on) and C[i] > sm[i]: on = True
        out[i] = on
    return out


def sim(sig, stop=1.0, trail=0.0):
    eq = 1.0; out = []; inp = False; ep = pk = np.nan; n = 0
    for i in range(1, len(C)):
        if inp: eq *= C[i] / C[i - 1]
        out.append(eq); w = bool(sig[i - 1])
        if inp:
            pk = max(pk, C[i])
            if (not w) or (stop < 1 and C[i] <= ep * (1 - stop)) or (trail > 0 and C[i] <= pk * (1 - trail)):
                inp = False; n += 1; eq *= 1 - FEE
        elif w:
            inp = True; ep = pk = C[i]; eq *= 1 - FEE
    return np.array(out), n


def idx(d): return max(1, int(np.searchsorted(DATES, np.datetime64(pd.Timestamp(d)))))


def met(eq, a, b=None):
    j = max(0, idx(a) - 1); e = eq[j:] if b is None else eq[j:max(0, idx(b) - 1)]
    d = DATES[1 + j:1 + j + len(e)]; r = np.diff(e) / e[:-1]
    yrs = (pd.Timestamp(d[-1]) - pd.Timestamp(d[0])).days / 365.25
    return dict(ret=e[-1] / e[0] - 1, mdd=(e / np.maximum.accumulate(e) - 1).min(),
                sh=r.mean() / (r.std() + 1e-12) * np.sqrt(252), cagr=(e[-1] / e[0]) ** (1 / yrs) - 1)


def row(name, sig, **kw):
    eq, n = sim(sig, **kw)
    cells = " | ".join(f"{(m := met(eq, a))['ret']:+.0%} {m['sh']:.2f} {m['mdd']:.0%}" for a in WINDOWS.values())
    print(f"| {name} | {cells} | {n} |")
    return eq


if __name__ == "__main__":
    print("| strategy | " + " | ".join(f"{k} ret/Sharpe/maxDD" for k in WINDOWS) + " | trades |")
    print("|---|" + "---|" * (len(WINDOWS) + 1))
    row("**live: MACD 10/20/9, −5% stop**", macd_sig(10, 20, 9), stop=0.05)
    row("MACD 12/15/14, −5% stop (best re-tune)", macd_sig(12, 15, 14), stop=0.05)
    row("crash-shield 8% / SMA20 (126d high)", crash(0.08, 20, 126))
    row("SMA200 ±2% band", ma_band(200, .02, .02))
    row("**SMA200 ±2% band + 12% trail (adopted)**", ma_band(200, .02, .02), trail=0.12)
    row("SMA200 ±2% | MACD 12/26/9 + 12% trail", ma_band(200, .02, .02) | macd_sig(12, 26, 9), trail=0.12)
    print("\nPlateau — full-history Sharpe, SMA window (rows) × symmetric band (cols 0,1,2,3,4%):")
    for n in (100, 125, 150, 175, 200, 225, 250):
        print(n, [round(float(met(sim(ma_band(n, b, b))[0], "2015-01-01")["sh"]), 2) for b in (0, .01, .02, .03, .04)])
    print("\nWalk-forward — pick by Sharpe on data before the cut, report the test window:")
    grid = [("band", (n, b, b), t) for n in (100, 125, 150, 175, 200, 225, 250)
            for b in (0, .01, .02, .03, .04) for t in (0, .12)]
    grid += [("macd", (f, s, g), 0) for f in (5, 8, 10, 12, 15) for s in (15, 20, 26, 35, 50)
             for g in (3, 5, 9, 14) if f < s]
    runs = [(k, p, t, sim(ma_band(*p) if k == "band" else macd_sig(*p),
                          stop=1.0 if k == "band" else .05, trail=t)[0]) for k, p, t in grid]
    for cut in ("2021-01-01", "2023-01-01"):
        for kind in ("band", "macd"):
            sub = [r for r in runs if r[0] == kind]
            k, p, t, eq = max(sub, key=lambda r: met(r[3], "2015-01-01", cut)["sh"])
            te = met(eq, cut); r25 = met(eq, "2025-01-01")
            print(f"cut {cut} {kind:4s} {p} trail {t}: test {te['ret']:+.0%} Sharpe {te['sh']:.2f} | 2025→ {r25['ret']:+.0%} Sharpe {r25['sh']:.2f}")
