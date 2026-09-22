"""Can the repo's downtrend signatures be traded SHORT profitably and reliably?

Every strategy in this repo is long/flat.  The same engines also emit bearish
reads — the CT divergence D1/D2/D3 "downtrend signatures" (TREND_SIGNATURES.md),
the ETF divergence D1/D2/D3, and the trend-family "below trend" state.  This
script asks whether any of them, turned into a SHORT position, earns a
reliable, risk-controlled profit after costs.

Signals are rebuilt with each app's own code (no reimplementation):
  * BTC / ETH / MSTR / MSTU — ``btc_ct_engine.compute_sigs_pure`` on the CT
    ensemble, prices from ``btc_ct_engine._load_prices`` (honest equity fills).
  * PBW / ARTY              — ``backtest_ticker.build_predictions`` +
    ``precompute_signals`` (ridge fit strictly pre-2021), lagged one bar.
  * GDX / NUGT              — ``backtest_gldm`` divergence on the GLDM signal.
  * SOXX/SOXL, GRID, REMX, WGMI, XLE/OIH/ERX, GLDM/UGL — ``trend_long_array``.

Convention for every panel row i: ``dn_*[i]`` is knowable at the moment
``px[i]`` can be filled, so a position decided at i earns px[i+1]/px[i]-1.

Offline: reads only the committed CSVs under data/.  Run:
    python scripts/eval_short_signals.py            # prints tables, writes CSVs
    python scripts/eval_short_signals.py --panels   # just list the signal panels
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

OUT = _REPO / "artifacts" / "short_signals_eval"
CT_OOS_START = "2026-03-01"      # CT ensemble calibration_meta.train_end = 2026-02-28
ETF_OOS_START = "2021-01-01"     # ridge models fit strictly before this date
RNG = np.random.default_rng(7)

# Annual short-borrow cost assumptions (conservative, IBKR-style general
# collateral for liquid ETFs; hard-to-borrow for MSTR and the leveraged ETFs).
# BTC/ETH are assumed shorted via perps: funding is usually PAID TO shorts in
# bull markets, so 0 is conservative-neutral.
BORROW = {"BTC": 0.0, "ETH": 0.0, "MSTR": 0.05, "MSTU": 0.20,
          "SOXX": 0.005, "SOXL": 0.03, "GRID": 0.02, "REMX": 0.02, "WGMI": 0.05,
          "XLE": 0.003, "OIH": 0.005, "ERX": 0.03, "GLDM": 0.005, "UGL": 0.03,
          "GDX": 0.003, "NUGT": 0.03, "PBW": 0.02, "ARTY": 0.03}
COST_SIDE = 0.001                # 10 bp per side: commission + slippage


# ════════════════════════════════════════════════════════════════════════
# Panel builders — one dict per traded instrument
# ════════════════════════════════════════════════════════════════════════
def _panel(key, family, dates, px, dn, bear, up, oos, ppy):
    return dict(key=key, family=family, dates=pd.DatetimeIndex(dates),
                px=np.asarray(px, float), dn=dn, bear=np.asarray(bear, bool),
                up=np.asarray(up, bool), oos=pd.Timestamp(oos), ppy=ppy)


def build_ct_panels() -> list[dict]:
    try:
        import sklearn._loss._loss as _sl
        sys.modules.setdefault("_loss", _sl)
    except Exception:
        pass
    import btc_ct_engine as E
    T = E.T
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    comp = T.prep(rf, preds, "2024-01-01", str(rf.index[-1].date()))
    dates = pd.DatetimeIndex(comp["target_date"])
    s = E.compute_sigs_pure(comp)
    px = E._load_prices(dates, comp)
    d1, d2, d3 = s["d1"], s["d2"], s["d3"]
    nd = d1.astype(int) + d2.astype(int) + d3.astype(int)
    dn = dict(D1=d1, D2=d2, D3=d3, HIGH_DN=nd >= 2, ANY_DN=nd >= 1,
              EXIT_DN=d3 | (d2 & ~s["bull_regime"]),
              D_BEAR=(nd >= 1) & ~s["above_ma30"])
    out = []
    for k in ("BTC", "ETH", "MSTR", "MSTU"):
        if k not in px:
            continue
        out.append(_panel(k, "ct-divergence", dates, px[k], dn,
                          ~s["above_ma30"], s["u1"], CT_OOS_START,
                          365 if k in ("BTC", "ETH") else 252))
    return out


def _div_arrays(sig, U1, D2, D1):
    u1 = (sig["ehma3"] > U1) & (sig["hb3"] >= 2)
    d2 = sig["ehma3"] < D2
    d1 = (sig["lb3"] >= 2) & (sig["elma3"] > D1)
    d3 = np.asarray(sig["d3"], bool)
    nd = d1.astype(int) + d2.astype(int) + d3.astype(int)
    dn = dict(D1=d1, D2=d2, D3=d3, HIGH_DN=nd >= 2, ANY_DN=nd >= 1,
              EXIT_DN=d2 | d3, D_BEAR=(nd >= 1) & ~np.asarray(sig["above_ma20"], bool))
    return dn, u1, ~np.asarray(sig["above_ma20"], bool)


def _cached_daily(key: str) -> pd.DataFrame:
    return pd.read_csv(_REPO / "data" / key.lower() / "macro_daily.csv",
                       index_col=0, parse_dates=True)


def build_etf_div_panels() -> list[dict]:
    import ticker_config as tcfg
    import backtest_ticker as bt
    out = []
    for key in ("PBW", "ARTY"):
        cfg = tcfg.CONFIGS[key]
        daily = _cached_daily(key)
        preds = bt.build_predictions(cfg, daily)
        sig = bt.lag_signals(bt.precompute_signals(cfg, preds))
        dn, u1, bear = _div_arrays(sig, cfg.u1_errhi_min, cfg.d2_errhi_max,
                                   cfg.d1_errlo_min)
        out.append(_panel(key, "etf-divergence", preds["target_date"],
                          preds["px_close"], dn, bear, u1, ETF_OOS_START, 252))
    return out


def build_gold_panels() -> list[dict]:
    import gldm_core as gc
    import backtest_gldm as bg
    daily = pd.read_csv(_REPO / "data" / "gldm" / "gldm_macro_daily.csv",
                        index_col=0, parse_dates=True)
    preds = bg.build_predictions(daily)
    sig = bg.lag_signals(bg.precompute_signals(preds))
    dn, u1, bear = _div_arrays(sig, gc.U1_ERRHI_MIN, gc.D2_ERRHI_MAX, gc.D1_ERRLO_MIN)
    out = []
    for k in ("GDX", "NUGT"):
        col = f"{k.lower()}_close"
        if col in preds:
            out.append(_panel(k, "gold-divergence", preds["target_date"], preds[col],
                              dn, bear, u1, ETF_OOS_START, 252))
    # GLDM / UGL trade the dual-MA 25/100 on the GLDM close
    g = daily["gldm_close"].astype(float)
    long_ = (g.rolling(25, min_periods=1).mean() > g.rolling(100, min_periods=1).mean()).to_numpy()
    for k in ("GLDM", "UGL"):
        col = f"{k.lower()}_close"
        if col in daily:
            out.append(_trend_panel(k, daily.index, daily[col], long_, g.to_numpy(float)))
    return out


def _trend_panel(key, dates, px, long_, sig_close):
    """Trend family: the bearish read is simply 'trend filter off'.  Also
    expose a stricter 'confirmed downtrend' (off AND close < 50d SMA AND the
    50d SMA falling) and a 'fresh cross' (first bar the filter flips off)."""
    long_ = np.asarray(long_, bool)
    c = pd.Series(np.asarray(sig_close, float))
    sma50 = c.rolling(50, min_periods=50).mean()
    falling = (sma50 < sma50.shift(10)).fillna(False).to_numpy()
    below = (c < sma50).fillna(False).to_numpy()
    off = ~long_
    fresh = off & np.r_[False, long_[:-1]]
    dn = dict(TREND_OFF=off, TREND_OFF_CONFIRMED=off & below & falling,
              FRESH_CROSS_DN=fresh)
    return _panel(key, "trend", dates, px, dn, off, long_, ETF_OOS_START, 252)


def build_trend_panels() -> list[dict]:
    import ticker_config as tcfg
    import backtest_ticker as bt
    out = []
    for key, sibs in (("SOXX", ["soxl"]), ("GRID", []), ("REMX", []),
                      ("WGMI", []), ("XLE", ["oih", "erx"])):
        cfg = tcfg.CONFIGS[key]
        daily = _cached_daily(key)
        gcl = daily["px_close"].astype(float).to_numpy()
        long_ = bt.trend_long_array(cfg, gcl)
        out.append(_trend_panel(key, daily.index, gcl, long_, gcl))
        for s in sibs:
            col = f"{s}_close"
            if col in daily:
                out.append(_trend_panel(s.upper(), daily.index,
                                        daily[col].ffill(), long_, gcl))
    return out


def build_all() -> list[dict]:
    panels = []
    for fn in (build_ct_panels, build_etf_div_panels, build_gold_panels,
               build_trend_panels):
        try:
            panels += fn()
        except Exception as exc:                      # keep the others going
            print(f"!! {fn.__name__} failed: {exc!r}")
    return panels



# ════════════════════════════════════════════════════════════════════════
# 1. Event study — is the forward return after a bearish read negative, and
#    more negative than a random bar?  Circular-shift permutation keeps the
#    signal's own clustering (the fair null for autocorrelated signals).
# ════════════════════════════════════════════════════════════════════════
HORIZONS = (1, 5, 10)


def _fwd(px, h):
    f = np.full(len(px), np.nan)
    f[:-h] = px[h:] / px[:-h] - 1
    return f


def event_study(p, n_perm=2000):
    rows = []
    for win, mask in (("IS", p["dates"] < p["oos"]), ("OOS", p["dates"] >= p["oos"])):
        for name, sig in p["dn"].items():
            sig = np.asarray(sig, bool)
            for h in HORIZONS:
                f = _fwd(p["px"], h)
                ok = mask & np.isfinite(f)
                s = sig & ok
                n = int(s.sum())
                if n < 5:
                    continue
                base = float(np.nanmean(f[ok]))
                cond = float(np.nanmean(f[s]))
                hit = float(np.mean(f[s] < 0))
                # permutation: rotate the signal inside the window
                idx = np.flatnonzero(ok)
                sv, fv = sig[idx], f[idx]
                shifts = RNG.integers(1, len(idx), n_perm)
                null = np.array([fv[np.roll(sv, k)].mean() for k in shifts])
                pval = float(np.mean(null <= cond))
                rows.append(dict(asset=p["key"], family=p["family"], window=win,
                                 signal=name, h=h, n=n, fwd_mean=cond * 100,
                                 base_mean=base * 100, edge=(base - cond) * 100,
                                 short_hit=hit * 100, p=pval))
    return rows


# ════════════════════════════════════════════════════════════════════════
# 2. Short-trade simulator — a real (non-rebalanced) short: P&L on the entry
#    notional, borrow accrued per bar, 10 bp per side, close-based buy-stop.
#    pos decided at bar i (signal knowable at px[i]) → filled at px[i].
# ════════════════════════════════════════════════════════════════════════
def sim_short(p, entry, cover, max_hold=None, stop=None, start=None, end=None):
    px, dates = p["px"], p["dates"]
    borrow_bar = BORROW.get(p["key"], 0.02) / p["ppy"]
    i0 = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start)))
    i1 = len(px) if end is None else int(dates.searchsorted(pd.Timestamp(end)))
    nav = 1.0; in_s = False; e_px = e_nav = 0.0; held = 0
    navs = np.full(i1 - i0, np.nan); pos = np.zeros(i1 - i0); trades = []
    for j, i in enumerate(range(i0, i1)):
        pr = px[i]
        if not np.isfinite(pr):
            navs[j] = navs[j - 1] if j else nav; pos[j] = float(in_s); continue
        if in_s:
            held += 1
            cur = e_nav * (1 - (pr / e_px - 1)) - e_nav * borrow_bar * held
            hit_stop = stop is not None and pr >= e_px * (1 + stop)
            if hit_stop or cover[i] or (max_hold and held >= max_hold):
                nav = cur - abs(cur) * COST_SIDE
                trades.append(nav / e_nav - 1)
                in_s = False
            else:
                navs[j] = cur; pos[j] = 1.0; continue
        if not in_s and entry[i] and not cover[i] and i < i1 - 1:
            nav -= nav * COST_SIDE
            in_s, e_px, e_nav, held = True, pr, nav, 0
            pos[j] = 1.0
        navs[j] = nav
    if in_s:
        trades.append(navs[-1] / e_nav - 1)
    return pd.Series(navs, index=dates[i0:i1]).ffill(), pos, np.array(trades)


def metrics(nav: pd.Series, ppy: int, pos=None, trades=None) -> dict:
    nav = nav.dropna()
    if len(nav) < 3:
        return {}
    r = nav.pct_change().dropna()
    yrs = max(len(nav) / ppy, 1e-9)
    tot = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if nav.iloc[-1] > 0 else -1
    sh = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1).min())
    out = dict(ret=tot * 100, cagr=cagr * 100, sharpe=float(sh), mdd=mdd * 100)
    if pos is not None:
        out["exposure"] = float(np.mean(pos)) * 100
    if trades is not None:
        out["trades"] = int(len(trades))
        out["win"] = float(np.mean(trades > 0) * 100) if len(trades) else np.nan
        out["worst_trade"] = float(trades.min() * 100) if len(trades) else np.nan
    return out


def shuffle_pvalue(p, entry, cover, kw, start, end, observed, n=300):
    """Null: same rule, same trade count/cluster shape, random timing —
    rotate the ENTRY array.  p = share of rotations with Sharpe ≥ observed."""
    idx = np.arange(len(entry))
    sh = []
    for k in RNG.integers(20, len(entry) - 20, n):
        e2 = np.roll(entry, k)
        nav, pos, tr = sim_short(p, e2, cover, start=start, end=end, **kw)
        sh.append(metrics(nav, p["ppy"]).get("sharpe", 0.0))
    return float(np.mean(np.array(sh) >= observed))


# Rule grid — every cell is reported; nothing is picked in-sample.
HOLDS = (5, 10, 20)
STOPS = (None, 0.08)


def short_grid(p):
    rows = []
    cover_up = np.asarray(p["up"], bool)           # bullish read covers
    for name, sig in p["dn"].items():
        entry = np.asarray(sig, bool)
        variants = [("hold%d" % h, dict(max_hold=h, stop=st), cover_up)
                    for h in HOLDS for st in STOPS]
        # state-short: stay short while the bear regime persists
        variants += [("state", dict(max_hold=None, stop=st), ~p["bear"] | cover_up)
                     for st in STOPS]
        for vname, kw, cover in variants:
            for win, st, en in (("IS", None, p["oos"]), ("OOS", p["oos"], None)):
                nav, pos, tr = sim_short(p, entry, cover, start=st, end=en, **kw)
                m = metrics(nav, p["ppy"], pos, tr)
                if not m or m.get("trades", 0) == 0:
                    continue
                rows.append(dict(asset=p["key"], family=p["family"], signal=name,
                                 rule=vname, stop=kw["stop"] or 0, window=win, **m))
    return rows


# ════════════════════════════════════════════════════════════════════════
# 3. Long/short overlay on the EXISTING long/flat trend strategies:
#    long while trend on, short (with stop) while trend off.
# ════════════════════════════════════════════════════════════════════════
def _daily_nav(p, pos_signed, start, borrow=True):
    px = p["px"]; d = p["dates"]
    r = np.r_[0.0, px[1:] / px[:-1] - 1]
    r = np.nan_to_num(r)
    held = np.r_[0.0, pos_signed[:-1]]             # decided at i-1, earns bar i
    turn = np.abs(np.diff(np.r_[0.0, held]))
    bor = (held < 0) * BORROW.get(p["key"], 0.02) / p["ppy"] if borrow else 0.0
    dr = held * r - turn * COST_SIDE - bor
    i0 = int(d.searchsorted(pd.Timestamp(start)))
    nav = pd.Series(np.cumprod(1 + dr[i0:]), index=d[i0:])
    return nav, held[i0:]


def overlay(p):
    if p["family"] != "trend":
        return []
    rows = []
    on = np.asarray(p["up"], bool)
    for win, start in (("full-OOS", p["oos"]),):
        for label, pos in (("long/flat (current)", on.astype(float)),
                           ("long/short", np.where(on, 1.0, -1.0)),
                           ("long/short-confirmed",
                            np.where(on, 1.0, np.where(p["dn"]["TREND_OFF_CONFIRMED"], -1.0, 0.0))),
                           ("buy & hold", np.ones(len(on)))):
            nav, held = _daily_nav(p, pos, start)
            rows.append(dict(asset=p["key"], variant=label, window=win,
                             **metrics(nav, p["ppy"], np.abs(held))))
    return rows


def ct_overlay(panels):
    """BTC-family: the live long/flat engine's position + a short leg on
    EXIT_DN held until U1 / above-MA30 (the best-motivated CT short rule)."""
    import btc_ct_engine as E
    res = {r["key"]: r for r in E.run_btc_ct()}
    rows = []
    for p in panels:
        if p["family"] != "ct-divergence" or p["key"] not in res:
            continue
        lp = res[p["key"]]["pos_series"].reindex(p["dates"]).fillna(0).to_numpy()
        _, spos, _ = sim_short(p, np.asarray(p["dn"]["EXIT_DN"], bool),
                               ~p["bear"] | np.asarray(p["up"], bool),
                               max_hold=10, stop=0.08)
        spos = np.where(lp > 0, 0.0, spos)
        for win, start in (("IS", p["dates"][35]), ("OOS", p["oos"])):
            for label, pos in (("long/flat (current)", lp),
                               ("long + EXIT_DN short", lp - spos),
                               ("buy & hold", np.ones(len(lp)))):
                nav, held = _daily_nav(p, pos, start)
                if win == "IS":
                    nav = nav[nav.index < p["oos"]]; held = held[:len(nav)]
                rows.append(dict(asset=p["key"], variant=label, window=win,
                                 **metrics(nav, p["ppy"], np.abs(held))))
    return rows


# ════════════════════════════════════════════════════════════════════════
# 4. Volatility study — do the bearish reads forecast SIZE rather than SIGN?
#    Forward 10-bar realised vol after the signal vs all bars (rotation null),
#    and how often the next 10 bars bring a >1σ move UP vs DOWN.
# ════════════════════════════════════════════════════════════════════════
def vol_study(p, n_perm=1000):
    px = p["px"]
    r = np.r_[np.nan, np.log(px[1:] / px[:-1])]
    fv = pd.Series(r).rolling(10).std().shift(-10).to_numpy()
    f10 = _fwd(px, 10)
    ok = np.isfinite(fv) & np.isfinite(f10)
    q = np.nanstd(f10[ok])
    rows = []
    for name, sig in p["dn"].items():
        sig = np.asarray(sig, bool); s = sig & ok
        if s.sum() < 10:
            continue
        idx = np.flatnonzero(ok); sv, fvv = sig[idx], fv[idx]
        null = np.array([fvv[np.roll(sv, k)].mean()
                         for k in RNG.integers(1, len(idx), n_perm)])
        rows.append(dict(asset=p["key"], family=p["family"], signal=name,
                         n=int(s.sum()), vol_ratio=float(fv[s].mean() / fv[ok].mean()),
                         p_vol=float(np.mean(null >= fv[s].mean())),
                         big_up=float(np.mean(f10[s] > q) * 100),
                         big_dn=float(np.mean(f10[s] < -q) * 100),
                         base_big_up=float(np.mean(f10[ok] > q) * 100),
                         base_big_dn=float(np.mean(f10[ok] < -q) * 100)))
    return rows


# ════════════════════════════════════════════════════════════════════════
def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    panels = build_all()
    ev = pd.DataFrame([r for p in panels for r in event_study(p)])
    grid = pd.DataFrame([r for p in panels for r in short_grid(p)])
    ov = pd.DataFrame([r for p in panels for r in overlay(p)] + ct_overlay(panels))
    vs = pd.DataFrame([r for p in panels for r in vol_study(p)])
    OUT.parent.mkdir(exist_ok=True)
    ev.to_csv(f"{OUT}.events.csv", index=False)
    grid.to_csv(f"{OUT}.grid.csv", index=False)
    ov.to_csv(f"{OUT}.overlay.csv", index=False)
    vs.to_csv(f"{OUT}.vol.csv", index=False)
    _report(ev, grid, ov, vs)
    print(f"wrote {OUT}.{{events,grid,overlay,vol}}.csv")


def _report(ev, grid, ov, vs):
    """Console summary — the numbers SHORT_SIGNALS_EVAL.md quotes."""
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\n── 1. Event study (h=5): short edge = base − post-signal fwd return, pp")
    e5 = ev[ev.h == 5]
    print(e5.groupby(["family", "window"]).agg(
        cells=("n", "size"), edge_pos=("edge", lambda x: (x > 0).mean()),
        significant=("p", lambda x: (x < 0.05).mean()),
        med_edge=("edge", "median")).round(2))
    print("\n── 2. Short-rule grid: share of cells profitable / Sharpe>0.5")
    print(grid.groupby(["family", "window"]).agg(
        cells=("ret", "size"), profitable=("ret", lambda x: (x > 0).mean()),
        sharpe_gt_half=("sharpe", lambda x: (x > 0.5).mean()),
        med_sharpe=("sharpe", "median"), med_mdd=("mdd", "median")).round(2))
    w = grid.pivot_table(index=["asset", "signal", "rule", "stop"],
                         columns="window", values="sharpe")
    both = w[(w.get("IS", 0) > 0.5) & (w.get("OOS", 0) > 0.5)]
    print(f"   cells with Sharpe>0.5 in BOTH IS and OOS: {len(both)} of {len(w)}")
    print("\n── 3. Overlay (adding the short leg to the live long/flat engine)")
    print(ov[["asset", "variant", "window", "ret", "sharpe", "mdd", "exposure"]]
          .round(2).to_string(index=False))
    print("\n── 4. Volatility: fwd-10 realised vol ratio and big-move asymmetry")
    print(vs.groupby("family").agg(
        cells=("n", "size"), med_vol_ratio=("vol_ratio", "median"),
        vol_sig=("p_vol", lambda x: (x < 0.05).mean()),
        big_up=("big_up", "median"), big_dn=("big_dn", "median")).round(2))


if __name__ == "__main__":
    if "--panels" in sys.argv:
        for p in build_all():
            sigs = {k: int(np.asarray(v).sum()) for k, v in p["dn"].items()}
            print(p["key"], p["family"], p["dates"][0].date(), p["dates"][-1].date(),
                  len(p["px"]), sigs)
    else:
        main()
