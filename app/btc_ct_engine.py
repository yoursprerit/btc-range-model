"""BTC / MSTR / MSTU / ETH via the BTC app's ACTUAL signal engine.

The BTC app (``app/btc_hourly_app.py``) is not importable (it runs Streamlit at
module top), but its H/L prediction + Pure-Regime backtest are byte-identically
reproduced by the importable research module ``backtest_trailing_stop`` (which
loads the trained CT ensemble ``models/inference_assets_ct.joblib`` and builds
the same 116-feature matrix — price + macro + Bitcoin on-chain + Coinbase
premium).  This module wraps it with the app's *live* entry gate and
thresholds (U1 err_hi>+1.3% + ≥2 high-breaks, D2 exit<−1.3%, MA30 regime gate,
regime-adaptive exit, per-asset stops, SL re-entry) so the Overall app's BTC
sleeve matches the BTC app in both live signal and back-test.  2026-07 retune:
all three assets trade the Standard MA (above-MA30) gate — the most profitable
and most stable gate on the current data.

2026-07-25 look-ahead fix: MSTR/MSTU fills moved from the *signal-bar date's*
equity close (printed ~15 h — up to 2.5 days over weekends — BEFORE the CT
signal is knowable at 12:00 UTC the next day) to the first exchange close at
or after signal availability (see ``_next_session_close`` and
ETH_BMNR_STRATEGY_EVAL.md §5).  On the 2026-07-25 vintage this moves MSTR
+296%→+165% and MSTU +685%→+499% over the full CT window; BTC/ETH are
unchanged (their 12:00-UTC bars fill exactly at the signal moment).  All
figures remain data-vintage-dependent.  Gate/stop config: all sleeves trade
the Standard MA gate; MSTR signal-exit-only, MSTU −6% stop, post-stop
re-entry override (MSTU only), 5-bar V-reversal window.

2026-07f: ETH (spot Ethereum) added as a fourth sleeve on the same parent BTC
signal — Standard-MA gate, signal-exit-only, no fixed stop (the MSTR
treatment).  ETH bars share BTC's 12:00-UTC anchor, so the sleeve's same-bar
fill lands exactly when the signal bar closes, and its history covers the whole
CT window (no staggered start).  ETH is the *signal* asset; live execution runs
through the ETHA ETF, exactly as the BTC sleeve executes via IBIT (see
scripts/ibkr_symbols.py).  See ETH_BMNR_STRATEGY_EVAL.md for the evaluation.

Caveat: the CT feature data (``data/backtest/raw_features_daily.csv``) spans
~2023-11 → the last pull, so BTC-sleeve returns begin ~2024 (not 2021).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_APP_DIR = Path(__file__).resolve().parent
_REPO = _APP_DIR.parent
for _p in (str(_REPO), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:                                            # CT model pickle needs this shim
    import sklearn._loss._loss as _sl
    sys.modules.setdefault("_loss", _sl)
except Exception:
    pass

import numpy as np
import pandas as pd

import backtest_trailing_stop as T             # loads the CT model at import

# ── self-heal the CT feature data so the BTC sleeve updates daily ────────────
# BTC/MSTR/MSTU signals are built from a committed daily-features CSV
# (data/backtest/raw_features_daily.csv).  In a long-running deployment nothing
# refreshes it, so those three freeze at the CSV's last date while every other
# instrument fetches live.  When the CSV falls behind the latest completed CT
# bar we re-pull it (scripts/pull_backtest_data.py) — at most once every few
# hours, and fully non-fatal: any failure just keeps the existing CSV.
_PULL_SCRIPT = _REPO / "scripts" / "pull_backtest_data.py"
_FEATURES_CSV = T.DATA / "raw_features_daily.csv"
_PULL_GUARD = T.DATA / ".last_feature_pull"
_PULL_MIN_GAP = pd.Timedelta(hours=3)          # don't re-attempt more often than this


def _features_last_bar() -> "pd.Timestamp | None":
    try:
        idx = pd.read_csv(_FEATURES_CSV, index_col=0, parse_dates=True).index
        return pd.Timestamp(idx[-1]).tz_localize(None).normalize()
    except Exception:
        return None


def _ensure_fresh_features() -> None:
    try:
        if not _FEATURES_CSV.exists() or not _PULL_SCRIPT.exists():
            return
        now = pd.Timestamp.utcnow().tz_localize(None)
        today = now.normalize()
        # the CT "day" is anchored at 12:00 UTC; the prior day's bar is complete
        # once we're past that anchor, so that's the freshest bar we can expect.
        target = today if now.hour >= 12 else (today - pd.Timedelta(days=1))
        last = _features_last_bar()
        if last is not None and last >= target:
            return                                 # already current — nothing to do
        if _PULL_GUARD.exists():                    # rate-limit re-attempts
            try:
                if (now - pd.Timestamp(_PULL_GUARD.read_text().strip())) < _PULL_MIN_GAP:
                    return
            except Exception:
                pass
        _PULL_GUARD.write_text(now.isoformat())     # stamp before running
        import subprocess
        # Fire-and-forget — NEVER block the Streamlit render on a network pull.
        # This bar's render uses the existing committed CSV; the detached pull
        # freshens it for a later render. A synchronous subprocess.run() here
        # blocked the first render for up to its timeout (~6 min) on a cold boot
        # with stale data — long enough that Streamlit Cloud's boot health-check
        # gave up and the app was stuck "in the oven" / crash-looped.
        subprocess.Popen([sys.executable, str(_PULL_SCRIPT)], cwd=str(_REPO),
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception:
        pass


# live thresholds (btc_hourly_app.py) and per-asset entry gate + stop.
# 2026-07 retune: all three assets use the Standard MA (above-MA30) gate — the
# most profitable and most stable gate on the current data (mirrors
# BTC/MSTR/MSTU_STRATEGY_GATE in btc_hourly_app.py). Only the stop differs.
# 2026-07e per-asset STOP retune (k-fold + synthetic-OOS validated):
#   BTC  → no fixed stop      (1× core; signal exits only)
#   MSTR → no fixed stop      (its −3% was ~0.6σ but only ever whipsawed in bulls;
#                              removing it lifts Full +212→+266%, Sharpe 1.34→1.41,
#                              MDD unchanged −22%, win-rate 75→85% — and a 1× name
#                              has no wipeout tail, so a stop adds nothing)
#   MSTU → fixed −6%          (2× fund; its −3% sat at only ~0.3σ and stopped out on
#                              routine noise. Vol-matched −6% (≈2× MSTR's old 3%)
#                              lifts Full +309→+524%, Sharpe 1.08→1.27, MDD −48→−42%,
#                              win-rate 33→62%, while still truncating the crash tail
#                              that going stopless blows open. See stop-loss eval.)
U1_ERRHI_MIN = 1.3
D2_ERRHI_MAX = -1.3
# 2026-07f — ETH (spot Ethereum) added as a fourth sleeve on the SAME parent BTC
# signal, with the MSTR treatment: Standard-MA gate, signal-exit-only, no fixed
# stop.  Evaluated in ETH_BMNR_STRATEGY_EVAL.md — a fixed stop ≤8% only hurts
# (the signal exits already cap intra-trade pain), exactly as on MSTR.  ETH is
# the signal/backtest asset and is executed live through the ETHA ETF.
STOP_PCT = {"BTC": None, "MSTR": None, "MSTU": 0.06,   # BTC/MSTR: no fixed stop; MSTU −6%
            "ETH": None}                               # ETH: MSTR treatment — no stop
GATE_BY_ASSET = {"BTC": "above_ma30", "MSTR": "above_ma30", "MSTU": "above_ma30",
                 "ETH": "above_ma30"}
# 2026-07 structural fix — post-stop re-entry override (STOPPED leveraged sleeve only).
# Within this many bars of a fixed-stop exit, a fresh U1 above the MA30 re-admits
# even when the XOR combined-block is on. Fixes the "stopped out at the
# capitulation low, then locked out of the recovery" failure. It fires only after a
# fixed-stop exit, so with the 2026-07e retune it applies to MSTU alone — BTC and
# MSTR now carry no stop (from_sl is never set), leaving them untouched. Stable 12–20 bars.
REENTRY_OVERRIDE_BARS = 12
# 2026-07c — V-reversal recency window (bars). Bridges the capitulation bar to the
# U1 confirmation a few bars later; widened 3→5 so the bridge survives data-vintage
# revisions (the Apr-2025 rally was missed on the live vintage when the old 3-bar
# window expired before U1 fired). Must match btc_hourly_app.V_RECENT_WIN.
V_RECENT_WIN = 5
_META = {
    "BTC":  dict(name="Bitcoin",       kind="core", stop=0.0),
    "MSTR": dict(name="MicroStrategy",  kind="beta", stop=0.0),
    "MSTU": dict(name="2× MSTR",        kind="lev",  stop=0.06),
    "ETH":  dict(name="Ethereum",       kind="core", stop=0.0),
}
ACCENT = "#f7931a"
EMOJI = "₿"


# ── signals: both live entry gates (U1>1.3 / D2<-1.3) ────────────────────
#   tf2_entry_pure = Pure Regime (OR gate)   — used for BTC
#   tf2_entry_ma   = Standard MA (XOR gate)  — used for MSTR/MSTU
def compute_sigs_pure(comp: pd.DataFrame) -> dict:
    """Byte-identical to the BTC app's run_*_backtest signal arrays, exposing
    both live entry gates (Pure Regime + Standard MA) and current thresholds."""
    N = len(comp)
    c = comp["close_asof"].values.astype(float)
    ph = comp["pred_high"].values.astype(float)
    pl = comp["pred_low"].values.astype(float)
    ah = comp["actual_high"].values.astype(float)
    al = comp["actual_low"].values.astype(float)
    err_hi = (ah - ph) / c * 100
    err_lo = (pl - al) / c * 100
    hi = (ah > ph).astype(int)
    lo = (al < pl).astype(int)
    ehma3 = np.zeros(N); elma3 = np.zeros(N)
    hb3 = np.zeros(N, int); lb3 = np.zeros(N, int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = err_hi[s:i + 1].mean(); elma3[i] = err_lo[s:i + 1].mean()
        hb3[i] = hi[s:i + 1].sum(); lb3[i] = lo[s:i + 1].sum()
    u1 = (ehma3 > U1_ERRHI_MIN) & (hb3 >= 2)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < D2_ERRHI_MAX
    d3 = np.zeros(N, bool)
    for i in range(1, N):
        cc = 0
        for k in range(i - 1, -1, -1):
            if hi[k]:
                cc += 1
            else:
                break
        if cc >= 3 and lo[i]:
            d3[i] = True
    ma30 = np.array([c[max(0, i - min(30, i + 1) + 1):i + 1].mean() for i in range(N)])
    above = c > ma30
    slope = np.zeros(N, bool)
    for i in range(5, N):
        slope[i] = ma30[i] > ma30[i - 5]
    bull = above & slope
    clean = np.array([not bool((d1[max(0, i - 7):i] | d2[max(0, i - 7):i]).any())
                      for i in range(N)])
    roll = np.array([err_hi[max(0, i - 29):i + 1].mean() for i in range(N)])
    dn = np.zeros(N)
    for i in range(N):
        nrm = max(abs(roll[i]), 0.01)
        dn[i] = ((-ehma3[i] / nrm) * .30 + (lb3[i] / 3.) * .30
                 + (elma3[i] / max(abs(elma3[i]), .10)) * .20 + float(lo[i]) * .20)
    vbar = (dn > 0.8) & (err_lo > 3.0)
    v = np.array([bool(vbar[max(0, i - (V_RECENT_WIN - 1)):i + 1].any()) for i in range(N)])
    tf_pure = u1 & (bull | (clean & ~above) | v)        # PURE-REGIME (OR gate)  — BTC
    tf_ma   = u1 & ((above ^ clean) | v)                # STANDARD MA (XOR gate) — MSTR/MSTU
    return dict(d1=d1, d2=d2, d3=d3, u1=u1, bull_regime=bull,
                tf2_entry=tf_pure, tf2_entry_pure=tf_pure, tf2_entry_ma=tf_ma,
                above_ma30=above, clean_7d=clean, v_recent=v,
                err_hi=err_hi, err_lo=err_lo, ehma3=ehma3, elma3=elma3,
                hb3=hb3, lb3=lb3)


# ── backtest loop — faithful copy of backtest_trailing_stop.run_bt, extended
#    to also return the per-bar position series and the open-position entry ──
def _run_bt(dates, asset_px, sigs, fixed_pct, bt_start):
    d2 = sigs["d2"]; d3 = sigs["d3"]; bull = sigs["bull_regime"]
    tf_entry = sigs["tf2_entry"]; above = sigs["above_ma30"]; vr = sigs["v_recent"]
    u1 = sigs["u1"]
    N = len(dates)
    WARMUP = T.WARMUP
    _bt0 = max(WARMUP, int(pd.DatetimeIndex(dates).searchsorted(bt_start))) \
        if bt_start is not None else WARMUP
    cap = T.INITIAL_CAPITAL
    nav = cap; pos = "CASH"; qty = 0.0
    e_price = e_nav = e_date = e_trig = None
    from_sl = False; bars_since_sl = 0; hwm = 0.0
    trades = []; nav_arr = np.full(N, np.nan); pos_arr = np.zeros(N)
    no_stop = (fixed_pct is None)
    fp = 10.0 if no_stop else float(fixed_pct)       # huge pct ⇒ hard stop never fires

    for i in range(N):
        price = asset_px[i]
        if i < _bt0:
            nav_arr[i] = cap; continue
        if not np.isfinite(price) or price <= 0:
            nav_arr[i] = qty * asset_px[i - 1] if pos == "LONG" and i > 0 else nav
            pos_arr[i] = 1.0 if pos == "LONG" else 0.0
            continue
        if pos == "LONG":
            cur = qty * price
            hwm = max(hwm, price)
            hard_stop = e_price * (1.0 - fp)
            fired_hard = (not no_stop) and price <= hard_stop
            sig_exit = bool(d3[i] or (d2[i] and not bull[i]))
            exit_lbl = "D3 exhaustion" if d3[i] else "D2 fade (bear)"
            if fired_hard:
                nav = qty * price
                trades.append(dict(entry_date=e_date, entry_price=e_price,
                                   exit_date=dates[i], exit_price=price,
                                   ret=price / e_price - 1, reason="stop −%.0f%%" % (fp * 100),
                                   duration_days=(dates[i] - e_date).days, was_sl=True))
                pos = "CASH"; qty = 0.0; from_sl = True; bars_since_sl = 0; hwm = 0.0
            elif sig_exit:
                nav = cur
                trades.append(dict(entry_date=e_date, entry_price=e_price,
                                   exit_date=dates[i], exit_price=price,
                                   ret=price / e_price - 1, reason=exit_lbl,
                                   duration_days=(dates[i] - e_date).days, was_sl=False))
                pos = "CASH"; qty = 0.0; from_sl = False; hwm = 0.0
            else:
                nav = cur
        else:
            if from_sl:
                bars_since_sl += 1
            _exit_at_i = d3[i] or (d2[i] and not bull[i])
            _reentry_ok = (not from_sl) or bool(bull[i]) or bars_since_sl >= 10
            # Post-stop re-entry override (see REENTRY_OVERRIDE_BARS): within the
            # window after a stop, a fresh U1 above the MA30 re-admits even if the
            # XOR combined-block would otherwise veto — recovers post-capitulation
            # rallies on the stopped (leveraged) sleeves. from_sl is only ever True
            # for MSTR/MSTU (BTC has no stop), so BTC is untouched.
            _override = bool(REENTRY_OVERRIDE_BARS and from_sl
                             and bars_since_sl <= REENTRY_OVERRIDE_BARS
                             and u1[i] and above[i] and not _exit_at_i)
            if (tf_entry[i] and _reentry_ok and (from_sl or not _exit_at_i)) or _override:
                qty = nav / price; e_price = price; e_date = dates[i]
                e_nav = nav; pos = "LONG"; hwm = price
                from_sl = False; bars_since_sl = 0
                e_trig = ("U1 + V-reversal" if vr[i]
                          else "U1 + ↑MA30" if above[i] else "U1 + Clean-7d")
        nav_arr[i] = qty * price if pos == "LONG" else nav
        pos_arr[i] = 1.0 if pos == "LONG" else 0.0

    if pos == "LONG" and np.isfinite(asset_px[N - 1]):
        nav_arr[N - 1] = qty * asset_px[N - 1]
    idx = pd.DatetimeIndex(dates[_bt0:])
    nav_s = pd.Series(nav_arr[_bt0:], index=idx).ffill()
    # ffill: equity sleeves can carry a NaN tail (signal bars whose next
    # exchange session hasn't printed yet — pending fills)
    bh_s = pd.Series(cap * asset_px[_bt0:] / asset_px[_bt0], index=idx).ffill()
    pos_s = pd.Series(pos_arr[_bt0:], index=idx)
    open_entry = (dict(price=float(e_price), date=pd.Timestamp(e_date), trigger=e_trig)
                  if pos == "LONG" else None)
    return dict(trades=trades, nav=nav_s, bh=bh_s, pos=pos_s,
                open_pos=(pos == "LONG"), open_entry=open_entry)


def _next_session_close(s: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Align an exchange close series onto the CT signal bars WITHOUT look-ahead.

    The CT bar labeled T closes at 12:00 UTC on calendar day T+1 — only then is
    bar T's signal knowable.  The first price an equity trader can actually get
    on that signal is therefore the close of the first exchange session on or
    after day T+1 (Monday for weekend bars).  The old ``reindex(dates).ffill()``
    filled at day T's close — ~15 h (up to 2.5 days over weekends) BEFORE the
    signal existed, systematically banking the correlated overnight/weekend gap
    (see ETH_BMNR_STRATEGY_EVAL.md §5).  Bars whose next session hasn't traded
    yet are NaN (a genuinely pending fill — the loop holds until it prints)."""
    want = pd.DatetimeIndex(dates) + pd.Timedelta(days=1)
    filled = s.reindex(s.index.union(want)).bfill()
    return filled.reindex(want).to_numpy(float)


def _load_prices(dates: pd.DatetimeIndex, comp: pd.DataFrame) -> dict:
    btc = comp["actual_close"].values.astype(float)
    mstr = _next_session_close(T.load_asset("MSTR"), dates)
    try:
        syn = pd.read_csv(T.DATA / "mstu_synthetic_daily.csv", parse_dates=["Date"])
        syn = syn.set_index("Date").sort_index()["close"].astype(float)
        mstu = _next_session_close(syn, dates)
    except Exception:
        mstu = _next_session_close(T.load_asset("MSTU"), dates)
    out = {"BTC": btc, "MSTR": mstr, "MSTU": mstu}
    # ETH spot on BTC's own 12:00-UTC bars — full coverage of the CT window, so
    # (unlike an ETF sleeve) there is no staggered start to work around.
    #
    # This load is NON-FATAL by design.  It used to be a bare
    # ``T.load_asset("ETH")``, so a missing/unreadable eth_usd_daily.csv (a
    # checkout that predates the file, or a failed background feature pull on a
    # long-running deployment) raised FileNotFoundError out of run_btc_ct() and
    # took down the WHOLE BTC app — BTC, MSTR and MSTU included.  The Overall
    # app then reported "⛔ app failed to load" and raised its flashing
    # STALE-SIGNALS alert even though the BTC signal itself was perfectly fresh.
    # One optional sibling must never be able to kill its parent's sleeves, so:
    #   1. prefer the 12:00-UTC CSV (correct bar anchor, matches the signal bar);
    #   2. else fall back to the CT feature matrix's own ``eth_close`` column,
    #      which is always present — note it is a midnight-UTC yfinance close, a
    #      DEGRADED anchor, so the sleeve's fills no longer coincide exactly with
    #      the signal bar;
    #   3. else omit ETH entirely and let the other three sleeves load.
    try:
        out["ETH"] = T.load_asset("ETH").reindex(dates).ffill().to_numpy(float)
    except Exception as exc:
        try:
            rf = T.load_raw_features()
            if "eth_close" not in rf.columns:
                raise KeyError("eth_close")
            # midnight-UTC closes: day T's close prints 12 h BEFORE bar T's
            # signal exists, so take the NEXT day's close (first one after the
            # 12:00-UTC signal moment) rather than ffilling a pre-signal print.
            out["ETH"] = (rf["eth_close"].astype(float)
                          .reindex(pd.DatetimeIndex(dates) + pd.Timedelta(days=1))
                          .to_numpy(float))
            warnings.warn(
                f"eth_usd_daily.csv unavailable ({exc}); ETH sleeve falling back to "
                "raw_features eth_close (midnight-UTC anchor — degraded fills).",
                RuntimeWarning)
        except Exception as exc2:
            warnings.warn(
                f"ETH sleeve unavailable ({exc}; fallback: {exc2}) — running "
                "BTC/MSTR/MSTU only so the BTC app still loads.", RuntimeWarning)
    return out


# ── result assembly (standard Overall-app result dicts) ──────────────────
_TRADING_DAYS = 252
CAP_BY_KIND = {"core": 0.30, "beta": 0.18, "lev": 0.10}
KIND_EMOJI = {"core": "", "beta": "⚡", "lev": "🔺"}


def _curve_metrics(equity: pd.Series) -> dict:
    eq = equity.to_numpy(float); idx = equity.index
    if len(eq) < 2:
        return dict(total_ret=0.0, cagr=0.0, mdd=0.0, sharpe=0.0, vol=0.0)
    total = eq[-1] / eq[0] - 1
    yrs = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1))
    rets = np.diff(eq) / eq[:-1]
    vol = float(np.std(rets) * np.sqrt(_TRADING_DAYS))
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(_TRADING_DAYS))
    return dict(total_ret=float(total), cagr=float(cagr), mdd=mdd, sharpe=sharpe, vol=vol)


def _decision(sigs, i, in_pos):
    d2 = bool(sigs["d2"][i]); d3 = bool(sigs["d3"][i]); bull = bool(sigs["bull_regime"][i])
    u1 = bool(sigs["u1"][i]); entry = bool(sigs["tf2_entry"][i])
    exit_sig = d3 or (d2 and not bull)                  # regime-adaptive (app logic)
    why = "D3 exhaustion" if d3 else "D2 fade (bear)"
    if in_pos:
        if exit_sig:
            return dict(state="EXIT", label=f"EXIT — {why}", ico="🔴", tone="exit")
        return dict(state="HOLD", label="LONG — HOLDING", ico="🟢", tone="hold")
    if exit_sig:
        return dict(state="AVOID", label=f"STAND ASIDE — EXIT ACTIVE ({why})",
                    ico="🟠", tone="watch")
    if entry:
        return dict(state="ENTRY", label="ENTER — STRATEGY BUY", ico="🟢", tone="buy")
    if u1:
        return dict(state="WATCH", label="WATCH — U1 (GATE PENDING)", ico="🟡", tone="watch")
    return dict(state="FLAT", label="FLAT — NO SIGNAL", ico="⬜", tone="flat")


def _alert(sigs, i, in_pos):
    if sigs["d3"][i] or (sigs["d2"][i] and not sigs["bull_regime"][i]):
        return "EXIT_DN"
    if sigs["tf2_entry"][i]:
        return "STRATEGY_BUY"
    if sigs["u1"][i]:
        return "WATCH_UP"
    return "NEUTRAL"


def run_btc_ct(start: str = "2024-01-01") -> list[dict]:
    """Run BTC/MSTR/MSTU through the BTC app's CT engine and return standard
    Overall-app result dicts (one per instrument)."""
    _ensure_fresh_features()          # refresh the feature CSV if it's behind
    rf = T.load_raw_features()
    preds = T.build_preds_offline(rf)
    end = str(rf.index[-1].date())
    comp = T.prep(rf, preds, start, end)
    dates = pd.DatetimeIndex(comp["target_date"])
    if len(dates) < 30:
        return []
    sigs = compute_sigs_pure(comp)
    px = _load_prices(dates, comp)
    sd = pd.Timestamp(start)
    last = len(dates) - 1
    bull_now = bool(sigs["bull_regime"][last])
    # cross-asset momentum (BTC distance above its 50-day SMA)
    bc = comp["actual_close"].to_numpy(float)
    ref = bc[max(0, len(bc) - 50):].mean()
    mom = (bc[-1] / ref - 1) if ref else 0.0
    as_of = pd.Timestamp(dates[last])

    out = []
    for key in [k for k in ("BTC", "MSTR", "MSTU", "ETH") if k in px]:
        meta = _META[key]
        # per-asset entry gate: BTC → Pure Regime, MSTR/MSTU/ETH → Standard MA
        sigs_k = dict(sigs)
        sigs_k["tf2_entry"] = (sigs["tf2_entry_ma"]
                               if GATE_BY_ASSET[key] == "above_ma30"
                               else sigs["tf2_entry_pure"])
        # start each sleeve at its asset's first finite bar, so a sleeve whose
        # series begins after the CT window opens contributes no phantom
        # pre-inception stream to B&H rebasing or the Overall matrices.  (All
        # four current sleeves cover the full window; this is a safety net.)
        _fin = np.isfinite(px[key]) & (px[key] > 0)
        sd_k = sd
        if _fin.any():
            sd_k = max(sd, pd.Timestamp(dates[int(np.argmax(_fin))]))
        bt = _run_bt(dates, px[key], sigs_k, STOP_PCT[key], sd_k)
        nav = bt["nav"]; bh = bt["bh"]
        ret = nav.pct_change().fillna(0.0).rename(key)
        pos_series = bt["pos"].rename(key)
        # r dict shaped like backtest_ticker.run_strategy for overall_core reuse
        # (this engine's trades ARE dicts with exit_date/ret, so they double as
        # the trade_log — without that key, overall_core's since-start trade
        # counter sees no closed trades for the BTC sleeves)
        r = dict(bh=bh.to_numpy(float), dates=list(nav.index), trades=bt["trades"],
                 trade_log=bt["trades"],
                 in_pos_now=bt["open_pos"], strat=nav.to_numpy(float))
        # display price = last FINITE fill: equity sleeves end with a NaN tail
        # (bars whose next exchange session hasn't printed yet)
        _fin_idx = np.flatnonzero(_fin)
        _last_fin = int(_fin_idx[-1]) if len(_fin_idx) else last
        last_px = float(px[key][_last_fin])
        prev_px = float(px[key][_last_fin - 1]) if _last_fin >= 1 else last_px
        dchg = (last_px / prev_px - 1) * 100 if prev_px else 0.0
        dec = _decision(sigs_k, last, bt["open_pos"])
        m = _curve_metrics(nav); bhm = _curve_metrics(bh)
        wr = 100.0 * np.mean([t["ret"] > 0 for t in bt["trades"]]) if bt["trades"] else 0.0
        pos = dict(in_pos=bt["open_pos"], entry_px=None, entry_date=None,
                   upnl=None, stop_px=None, days=None, dist_stop=None)
        if bt["open_pos"] and bt["open_entry"]:
            e_px = bt["open_entry"]["price"]; e_dt = bt["open_entry"]["date"]
            pos.update(entry_px=e_px, entry_date=e_dt,
                       upnl=(last_px / e_px - 1) * 100,
                       days=int((as_of - e_dt).days))
            if meta["stop"]:
                pos["stop_px"] = e_px * (1 - meta["stop"])
                pos["dist_stop"] = (last_px / pos["stop_px"] - 1) * 100
        last_trade = None
        if bt["trades"]:
            t = bt["trades"][-1]
            last_trade = dict(entry_date=t["entry_date"], exit_date=t["exit_date"],
                              entry_px=t["entry_price"], exit_px=t["exit_price"],
                              ret=t["ret"], reason=t["reason"])
        out.append(dict(
            key=key, parent="BTC", name=meta["name"], kind=meta["kind"],
            emoji=EMOJI, kemoji=KIND_EMOJI[meta["kind"]], accent=ACCENT,
            cap=CAP_BY_KIND[meta["kind"]],
            last_close=last_px, dchg=dchg, ma_val=None,
            sentiment=np.nan, decision=dec, alert=_alert(sigs_k, last, bt["open_pos"]),
            bull_regime=bull_now, pos=pos, last_trade=last_trade, mom=mom,
            metrics=m, bh_metrics=bhm, win_rate=wr, n_trades=len(bt["trades"]),
            ret=ret, pos_series=pos_series, strat=nav.to_numpy(float),
            dates=pd.Series(nav.index), r=r, as_of=as_of,
            mode="ct-divergence", ma_window=30,
            stop=meta["stop"], engine="ct",
        ))
    return out

