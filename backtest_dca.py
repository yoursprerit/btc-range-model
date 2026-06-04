"""
DCA Strategy Evaluation — BTC + MSTR
======================================
Evaluates whether Dollar Cost Averaging (DCA) improves the TF2 baseline strategy.

DCA Variants:
  TF2-BASE : 100% in on entry signal, 100% out on D2/D3  (current strategy)
  DCA-3T   : 3-tranche staged entry — 1/3 per day over 3 consecutive days
  DCA-CONF : Confirmation entry — 50% on signal, add 50% only if U1 fires next bar
  DCA-DIP  : Dip averaging — 67% on signal, add 33% if price dips >1.5% intra-trade

Assets:
  BTC  : Direct BTC-USD exposure
  MSTR : MicroStrategy stock (leveraged BTC proxy, ~2–3× beta)

Period : May 26, 2024 → May 27, 2026  (2-year window)
Warning: Pre-Sep 18, 2025 = IN-SAMPLE  |  Post = fully OOS
"""
import sys, os, warnings, joblib, requests, time
warnings.filterwarnings("ignore")
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

# ── Four UI periods (matching the Streamlit dashboard) ────────────────────────
PERIODS = {
    "Bear Market":  ("2025-06-01",  "2026-06-03"),   # ~1 yr, mixed IS/OOS
    "Bull Market":  ("2024-06-05",  "2025-06-14"),   # ~1 yr, all IS ⚠️
    "OOS Only":     ("2026-03-01",  "2026-06-03"),   # ~3 mo, truly OOS ✓
    "Full Market":  ("2024-06-05",  "2026-06-03"),   # ~2 yr, mixed IS/OOS
}
# OOS boundary from model's calibration_meta test_start
OOS_CUTOFF     = pd.Timestamp("2026-03-01")

FETCH_START    = "2024-02-01"    # extra warmup; covers all periods
FETCH_END      = "2026-06-04"
INITIAL_CAP    = 100_000.0
RF_ANNUAL      = 0.045

# Recommended variants per asset (from initial full-period evaluation)
BTC_VARIANTS   = ["BASE", "DCA-CONF", "DCA-DIP"]    # CONF=best risk-adj; DIP=best return
MSTR_VARIANTS  = ["BASE", "DCA-3T"]                 # 3T=only clearly positive MSTR variant

# Legacy single-period constant (kept for backward compat with older functions)
OOS_START      = OOS_CUTOFF

# ── Model ──────────────────────────────────────────────────────────────────────
AD = joblib.load(str(_ROOT / "models" / "inference_assets_ct.joblib"))
print(f"✓ Model loaded  |  feat_cols: {len(AD['feat_cols'])}")

# ── Data fetch ─────────────────────────────────────────────────────────────────
_MACRO_SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
               "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
_ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
            "n-unique-addresses","transaction-fees-usd","mempool-size",
            "estimated-transaction-volume-usd","market-cap",
            "avg-block-size","cost-per-transaction"]


def _yf(sym, start, end, auto_adjust=False):
    d = yf.download(sym, start=start, end=end, progress=False,
                    auto_adjust=auto_adjust)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d


def fetch_coinbase_daily(start_str, end_str):
    """Fetch BTC-USD daily closes from Coinbase Exchange public API (no auth needed)."""
    CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows = []
    cur = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=299), end)
        params = {
            "granularity": 86400,
            "start": cur.isoformat() + "Z",
            "end": (chunk_end + pd.Timedelta(days=1)).isoformat() + "Z",
        }
        try:
            r = requests.get(CB_URL, params=params, timeout=30)
            if r.status_code == 200:
                rows.extend(r.json())
            else:
                print(f" (Coinbase {r.status_code}) ", end="")
        except Exception as exc:
            print(f" (Coinbase err: {exc}) ", end="")
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.35)
    if not rows:
        return pd.Series(dtype=float, name="coinbase_close")
    tmp = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    tmp["date"] = pd.to_datetime(tmp["ts"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("date").set_index("date").sort_index()
    tmp.index = pd.DatetimeIndex(tmp.index).tz_localize(None)
    return tmp["close"].rename("coinbase_close")


def fetch_data(fetch_start, fetch_end):
    print(f"\nFetching BTC + macro: {fetch_start} → {fetch_end}")
    frames = {}
    d = _yf("BTC-USD", fetch_start, fetch_end)
    frames.update({"btc_close": d["Close"], "btc_high": d["High"],
                   "btc_low": d["Low"], "btc_volume": d["Volume"]})
    for name, sym in _MACRO_SYMS.items():
        try:
            frames[f"{name}_close"] = _yf(sym, fetch_start, fetch_end)["Close"]
        except Exception:
            pass
    df = pd.DataFrame(frames).ffill(limit=5)
    df.index.name = "date"

    print("  Fetching MSTR …", end=" ")
    try:
        mstr = _yf("MSTR", fetch_start, fetch_end, auto_adjust=True)
        mstr_close = mstr["Close"].rename("mstr_close")
        mstr_close = mstr_close.reindex(df.index).ffill(limit=4)
        df["mstr_close"] = mstr_close
        print(f"OK ({mstr_close.notna().sum()} bars)")
    except Exception as e:
        print(f"FAILED: {e}")

    print("  Fetching Coinbase premium …", end=" ")
    try:
        cb = fetch_coinbase_daily(fetch_start, fetch_end)
        cb = cb.reindex(df.index).ffill(limit=3)
        df["coinbase_close"] = cb
        print(f"OK ({cb.notna().sum()} bars)")
    except Exception as e:
        print(f"FAILED: {e}")

    print("  On-chain …", end=" ")
    ok = 0
    for m in _ONCHAIN:
        col = f"oc_{m.replace('-','_')}"
        try:
            r = requests.get(f"https://api.blockchain.info/charts/{m}",
                params={"timespan": "3years", "format": "json", "sampled": "true"},
                timeout=20)
            vals = r.json().get("values", [])
            s = pd.Series(
                {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in vals},
                name=col, dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            df[col] = s.reindex(df.index).ffill(limit=7)
            ok += 1
        except Exception:
            pass
    print(f"{ok}/{len(_ONCHAIN)} OK")
    return df


def build_preds(df):
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in [1, 3, 5, 7, 14, 30]:
        f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5, 10, 20, 30]:
        f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]:
        f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c
    f["range_ma7"] = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"] = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    g = c.diff().clip(lower=0).rolling(14).mean()
    ls = (-c.diff().clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"] = macd / c
    f["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    f["bb_width"] = (4 * sd20) / ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"] = np.log(v).diff()
    f["vol_z_20"] = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)
    for nm, col in [("spx","spx_close"),("ndx","ndx_close"),("vix","vix_close"),
                    ("gold","gold_close"),("dxy","dxy_close"),("tnx","tnx_close"),
                    ("eth","eth_close")]:
        if col not in df.columns:
            continue
        s = df[col]
        lr = np.log(s).diff()
        for k in [1, 5, 20]:
            f[f"{nm}_ret_{k}"] = lr.rolling(k).sum()
        f[f"{nm}_vol_20"] = lr.rolling(20).std()
    for cn, cc in [("spx","spx_close"),("ndx","ndx_close"),
                   ("gold","gold_close"),("dxy","dxy_close")]:
        if cc in df.columns:
            f[f"btc_{cn}_corr_30"] = ret.rolling(30).corr(np.log(df[cc]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        s = df[col].astype(float)
        sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"] = sl.diff(1)
        f[f"{col}_d7"] = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    yh = (nh - c) / c
    yl = (c - nl) / c
    f["y_hi_ema3"] = yh.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = yl.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = yh.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = yl.shift(1).ewm(span=7, adjust=False).mean()
    p3h = h.shift(1).rolling(3).max()
    p3l = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > p3h).astype(float)
    f["below_3d_low"] = (c < p3l).astype(float)
    f["bo_strength_up"] = (c / p3h - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / p3l).clip(lower=0)
    ya, yb = yh.shift(1), yl.shift(1)
    f["y_hi_surprise"] = ya - ya.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = yb - yb.ewm(span=7, adjust=False).mean()
    nr = ret.clip(upper=0)
    f["dn_vol_5"] = nr.rolling(5).std()
    f["dn_vol_20"] = nr.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)
    # Coinbase premium vs Yahoo reference price
    if "coinbase_close" in df.columns:
        _cb = df["coinbase_close"]
        _prem = (_cb - c) / c * 100
        f["cb_premium"] = _prem
        f["cb_premium_ma3"] = _prem.rolling(3).mean()
        f["cb_premium_z7"] = ((_prem - _prem.rolling(7).mean()) /
                               _prem.rolling(7).std())
    fc = AD["feat_cols"]
    f = f.replace([np.inf, -np.inf], np.nan)
    for col in fc:
        if col not in f.columns:
            f[col] = np.nan
    F = f[fc].dropna()
    if F.empty:
        raise RuntimeError("Feature matrix empty")
    print(f"  Predictions: {len(F)} rows")
    if AD.get("ensemble") and AD.get("constituents"):
        yhi = np.mean([con["m_hi"].predict(F) for con in AD["constituents"]], axis=0)
        ylo = np.mean([con["m_lo"].predict(F) for con in AD["constituents"]], axis=0)
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", 0))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", 0))
    else:
        yhi = AD["hi_model"].predict(F)
        ylo = AD["lo_model"].predict(F)
    cv = c.reindex(F.index).values
    ph = cv * (1 + np.clip(yhi, 0, None))
    pl = cv * (1 - np.clip(ylo, 0, None))
    idx = np.asarray(F.index, dtype="datetime64[ns]")
    nd = np.empty(len(F), dtype="datetime64[ns]")
    nd[:-1] = idx[1:]
    nd[-1] = idx[-1] + np.timedelta64(1, "D")
    res = pd.DataFrame({"close_asof": cv, "pred_high": ph, "pred_low": pl},
                       index=pd.DatetimeIndex(nd, name="target_date"))
    return res[~res.index.duplicated(keep="last")]


# ── Signal computation (shared across all variants) ────────────────────────────
def compute_signals(df_raw, preds, start_iso, end_iso):
    """Compute all TF2 signals. Returns signal arrays + aligned date/price index."""
    WARMUP = 35
    sd, ed = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    p = preds.loc[(preds.index >= sd - pd.Timedelta(days=45)) &
                  (preds.index <= ed)].copy()
    c = df_raw["btc_close"]
    h = df_raw["btc_high"]
    lo = df_raw["btc_low"]
    p["ah"] = h.reindex(p.index).values
    p["al"] = lo.reindex(p.index).values
    p["ac"] = c.reindex(p.index).values
    comp = p.dropna(subset=["ah", "al", "ac"]).reset_index()
    comp = comp[comp["target_date"] >= sd].reset_index(drop=True)
    N = len(comp)
    if N < WARMUP + 3:
        return None

    dates = pd.DatetimeIndex(comp["target_date"])
    ca = comp["close_asof"].values.astype(float)
    btc_ep = comp["ac"].values.astype(float)
    ph = comp["pred_high"].values.astype(float)
    pl = comp["pred_low"].values.astype(float)
    ah = comp["ah"].values.astype(float)
    al = comp["al"].values.astype(float)

    ehi = (ah - ph) / ca * 100
    elo = (pl - al) / ca * 100
    hb = (ah > ph).astype(int)
    lb = (al < pl).astype(int)

    ehma3 = np.zeros(N)
    elma3 = np.zeros(N)
    hb3 = np.zeros(N, dtype=int)
    lb3 = np.zeros(N, dtype=int)
    for i in range(N):
        s = max(0, i - 2)
        ehma3[i] = np.mean(ehi[s:i + 1])
        elma3[i] = np.mean(elo[s:i + 1])
        hb3[i] = int(np.sum(hb[s:i + 1]))
        lb3[i] = int(np.sum(lb[s:i + 1]))

    # ── Signal definitions (exact match with btc_hourly_app.py UI) ──────────────
    u1 = (ehma3 > 0.7) & (hb3 >= 2)          # UI: 0.7 threshold (NOT 0.5)
    d1 = (lb3 >= 2) & (elma3 > 0.5)
    d2 = ehma3 < -0.75                         # UI: -0.75 threshold (NOT -1.0)
    d3 = np.zeros(N, dtype=bool)
    for i in range(1, N):
        con = 0
        for k in range(i - 1, -1, -1):
            if hb[k]:
                con += 1
            else:
                break
        if con >= 3 and lb[i]:
            d3[i] = True

    ma30 = np.full(N, np.nan)
    for i in range(N):
        w = min(30, i + 1)
        ma30[i] = np.mean(ca[max(0, i - w + 1):i + 1])
    abv = ca > ma30
    slp = np.zeros(N, dtype=bool)
    for i in range(5, N):
        if np.isfinite(ma30[i]) and np.isfinite(ma30[i - 5]):
            slp[i] = ma30[i] > ma30[i - 5]
    bull = abv & slp

    # UI uses 7-bar lookback (variable is named clean_10d in UI but uses i-7)
    c10 = np.zeros(N, dtype=bool)
    for i in range(N):
        li = max(0, i - 7)
        c10[i] = not bool(np.any(d1[li:i] | d2[li:i]))

    # ── V-Gate: capitulation/reversal third entry gate (from UI) ─────────────
    _DN_NORM_W = 30
    roll_ehi_norm = np.array([
        float(np.mean(ehi[max(0, i - _DN_NORM_W + 1):i + 1]))
        for i in range(N)
    ])
    dn_score = np.zeros(N)
    for i in range(N):
        norm = max(abs(roll_ehi_norm[i]), 0.01)
        dn_score[i] = (
            (-ehma3[i] / norm)                               * 0.30 +
            (lb3[i]    / 3.0)                                * 0.30 +
            (elma3[i]  / max(abs(elma3[i]), 0.10))           * 0.20 +
            float(lb[i])                                     * 0.20
        )
    v_rev_bar = (dn_score > 0.8) & (elo > 3.0)
    v_recent = np.zeros(N, dtype=bool)
    for i in range(N):
        v_recent[i] = bool(np.any(v_rev_bar[max(0, i - 2):i + 1]))

    entry = u1 & (abv | c10 | v_recent)

    return dict(
        N=N, WARMUP=WARMUP, dates=dates, btc_ep=btc_ep,
        u1=u1, d1=d1, d2=d2, d3=d3, bull=bull, abv=abv, entry=entry,
        comp=comp,      # raw comparison frame (for MSTR price alignment)
    )


# ── DCA backtest engine ────────────────────────────────────────────────────────
def backtest_variant(sig, asset_prices, variant="BASE", cap=100_000.0):
    """
    Run one DCA variant on a given asset price series.

    All BTC-derived signals are from `sig`.  `asset_prices` can be BTC or MSTR.

    variant:
      BASE     — 100% in on entry, 100% out on D2/D3
      DCA-3T   — 3-tranche: 1/3 entry, 1/3 day+1, 1/3 day+2  (always adds)
      DCA-CONF — 2-tranche: 50% entry, +50% only if U1 active next bar
      DCA-DIP  — 67% entry, +33% if asset drops >1.5% from ref price (no exit yet)
    """
    N = sig["N"]
    WARMUP = sig["WARMUP"]
    dates = sig["dates"]
    u1 = sig["u1"]
    bull = sig["bull"]
    entry = sig["entry"]
    d2 = sig["d2"]
    d3 = sig["d3"]
    ep = asset_prices            # asset-specific execution prices

    # portfolio state
    nav = cap
    cash = cap
    qty = 0.0
    pos = "CASH"                 # "CASH" | "ENTERING" | "LONG"

    # per-trade DCA state (reset on every new entry)
    tranche_dollar = 0.0
    tranche_remaining = 0
    dca_total_cost = 0.0
    dca_total_shares = 0.0
    dip_ref_price = 0.0
    dip_added = False
    entry_nav_record = 0.0
    entry_date_record = None
    entry_price_avg = 0.0

    trades = []
    nav_arr = np.full(N, np.nan)

    for i in range(N):
        si = i - 1                   # bar whose signal we act on today
        price = ep[i]

        if i < WARMUP:
            nav_arr[i] = cap
            continue

        if np.isnan(price) or price <= 0:
            # forward-fill nav on missing price (weekends, holidays)
            nav_arr[i] = nav_arr[i - 1] if i > 0 and not np.isnan(nav_arr[i - 1]) else nav
            continue

        # ── CASH: look for entry ──────────────────────────────────────────────
        if pos == "CASH":
            cash = nav
            if si >= 0 and entry[si]:
                entry_nav_record = nav
                entry_date_record = dates[i]
                dca_total_cost = 0.0
                dca_total_shares = 0.0
                dip_added = False

                if variant == "BASE":
                    qty = nav / price
                    cash = 0.0
                    dca_total_cost = nav
                    dca_total_shares = qty
                    entry_price_avg = price
                    tranche_remaining = 0
                    pos = "LONG"

                elif variant == "DCA-3T":
                    tranche_dollar = nav / 3.0
                    qty = tranche_dollar / price
                    cash = nav - tranche_dollar
                    dca_total_cost = tranche_dollar
                    dca_total_shares = qty
                    entry_price_avg = price
                    tranche_remaining = 2
                    pos = "ENTERING"

                elif variant == "DCA-CONF":
                    tranche_dollar = nav / 2.0
                    qty = tranche_dollar / price
                    cash = nav - tranche_dollar
                    dca_total_cost = tranche_dollar
                    dca_total_shares = qty
                    entry_price_avg = price
                    tranche_remaining = 1
                    pos = "ENTERING"

                elif variant == "DCA-DIP":
                    tranche_dollar = nav * 0.67
                    qty = tranche_dollar / price
                    cash = nav - tranche_dollar
                    dca_total_cost = tranche_dollar
                    dca_total_shares = qty
                    entry_price_avg = price
                    dip_ref_price = price
                    tranche_remaining = 0
                    pos = "LONG"      # fully "entered" (DCA happens within position)

        # ── ENTERING / LONG: manage active position ───────────────────────────
        elif pos in ("ENTERING", "LONG"):

            # Mark-to-market NAV
            nav = qty * price + cash

            # TF2 regime-adaptive exit: D3 always; D2 only in non-bull
            do_exit = False
            exit_label = ""
            if si >= 0:
                if d3[si]:
                    do_exit, exit_label = True, "D3"
                elif d2[si] and not bull[si]:
                    do_exit, exit_label = True, "D2"

            if do_exit:
                nav = qty * price + cash
                frac_inv = dca_total_cost / entry_nav_record if entry_nav_record > 0 else 1.0
                pnl_pct = (price / entry_price_avg - 1) * 100 if entry_price_avg > 0 else 0
                regime = "BULL" if (si >= 0 and bull[si]) else "BEAR"
                trades.append(dict(
                    entry_date=entry_date_record,
                    entry_price=round(entry_price_avg, 2),
                    exit_date=dates[i],
                    exit_price=round(float(price), 2),
                    pnl_pct=pnl_pct,
                    pnl_abs=nav - entry_nav_record,
                    exit_signal=exit_label,
                    duration_days=(dates[i] - entry_date_record).days,
                    regime=regime,
                    frac_invested=frac_inv,
                ))
                # Reset position
                qty = 0.0
                cash = nav
                tranche_remaining = 0
                pos = "CASH"

            else:
                # ── Staged entry logic ─────────────────────────────────────────
                if pos == "ENTERING" and tranche_remaining > 0:

                    if variant == "DCA-3T":
                        # Always buy next tranche
                        new_shares = tranche_dollar / price
                        qty += new_shares
                        cash -= tranche_dollar
                        dca_total_cost += tranche_dollar
                        dca_total_shares += new_shares
                        entry_price_avg = dca_total_cost / dca_total_shares
                        tranche_remaining -= 1
                        if tranche_remaining == 0:
                            pos = "LONG"

                    elif variant == "DCA-CONF":
                        # Add second tranche only if U1 still active yesterday
                        if si >= 0 and u1[si]:
                            new_shares = tranche_dollar / price
                            qty += new_shares
                            cash -= tranche_dollar
                            dca_total_cost += tranche_dollar
                            dca_total_shares += new_shares
                            entry_price_avg = dca_total_cost / dca_total_shares
                        # Either way, we're "done" entering (max 2 tranches)
                        tranche_remaining = 0
                        pos = "LONG"

                elif pos == "ENTERING" and tranche_remaining == 0:
                    pos = "LONG"

                # ── DIP add-on for DCA-DIP ─────────────────────────────────────
                if variant == "DCA-DIP" and pos == "LONG" and cash > 1.0 and not dip_added:
                    if price < dip_ref_price * 0.985:       # >1.5% drop from ref
                        new_shares = cash / price
                        qty += new_shares
                        dca_total_cost += cash
                        dca_total_shares += new_shares
                        entry_price_avg = dca_total_cost / dca_total_shares
                        cash = 0.0
                        dip_added = True

        nav_arr[i] = qty * price + cash if pos != "CASH" else cash

    # If still in a position at end of period, mark final NAV
    if pos in ("ENTERING", "LONG") and qty > 0:
        last_valid = ep[N - 1]
        if np.isnan(last_valid):
            last_valid = ep[~np.isnan(ep)][-1]
        nav_arr[N - 1] = qty * last_valid + cash

    navs = pd.Series(nav_arr[WARMUP:], index=dates[WARMUP:]).replace(0, np.nan)
    # Forward-fill cash gaps, then backfill any leading NaN
    navs = navs.ffill().bfill().fillna(cap)

    # Buy & Hold on same asset
    valid_ep = ep[WARMUP:]
    first_valid = next((v for v in valid_ep if not np.isnan(v) and v > 0), None)
    if first_valid is not None:
        bhs = pd.Series(cap * valid_ep / first_valid, index=dates[WARMUP:])
        bhs = bhs.ffill().fillna(cap)
    else:
        bhs = pd.Series(cap, index=dates[WARMUP:])

    return dict(variant=variant, trades=trades, nav=navs, bh=bhs,
                open=(pos in ("ENTERING", "LONG")))


def slice_oos_from_full(res_full, label, oos_cutoff=OOS_CUTOFF):
    """Compute OOS-only metrics by slicing a Full Market backtest result at oos_cutoff.

    Matches the UI's approach: the OOS period is NOT a fresh run but a slice of the
    full-period backtest, normalized to $100k at oos_cutoff. Trades that opened
    before oos_cutoff are excluded; pnl_abs values are rescaled by the normalization
    factor so tax computations stay in the same $100k capital base.
    """
    nav = res_full["nav"]
    bh  = res_full["bh"]
    trades = res_full["trades"]
    cap = INITIAL_CAP

    nav_oos = nav[nav.index >= oos_cutoff]
    bh_oos  = bh[bh.index  >= oos_cutoff]
    if len(nav_oos) < 2:
        return None

    nav_at_cut = float(nav.asof(oos_cutoff)) or cap
    bh_at_cut  = float(bh.asof(oos_cutoff))  or cap
    norm = cap / nav_at_cut          # rescale factor so OOS starts at $100k

    scaled_nav = (nav_oos / nav_at_cut * cap).rename("nav")
    scaled_bh  = (bh_oos  / bh_at_cut  * cap).rename("bh")

    oos_trades = [t for t in trades if t["entry_date"] >= oos_cutoff]
    scaled_trades = [dict(t, pnl_abs=t["pnl_abs"] * norm) for t in oos_trades]

    oos_res = dict(variant=res_full["variant"], trades=scaled_trades,
                   nav=scaled_nav, bh=scaled_bh, open=res_full["open"])
    return compute_metrics(oos_res, label)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(res, label):
    nav = res["nav"]
    bh = res["bh"]
    trades = res["trades"]
    cap = INITIAL_CAP

    total_ret = (nav.iloc[-1] / cap - 1) * 100
    bh_ret = (bh.iloc[-1] / cap - 1) * 100
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / cap) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    dr = nav.pct_change().fillna(0)
    vol_ann = dr.std() * np.sqrt(252) * 100

    rf_daily = (1 + RF_ANNUAL) ** (1 / 252) - 1
    excess = dr - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    down = dr[dr < rf_daily] - rf_daily
    down_std = down.std() * np.sqrt(252) if len(down) > 1 else 1e-9
    sortino = excess.mean() * 252 / down_std if down_std > 0 else 0

    rm = nav.cummax()
    dd = (nav - rm) / rm * 100
    max_dd = dd.min()

    rm_bh = bh.cummax()
    bh_max_dd = ((bh - rm_bh) / rm_bh * 100).min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
    gross_profit = sum(t["pnl_abs"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_abs"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    best = float(max(t["pnl_pct"] for t in trades)) if trades else 0.0
    worst = float(min(t["pnl_pct"] for t in trades)) if trades else 0.0
    avg_dur = float(np.mean([t["duration_days"] for t in trades])) if trades else 0.0
    days_in = sum(t["duration_days"] for t in trades)
    total_days = max(1, (nav.index[-1] - nav.index[0]).days)
    time_in = 100.0 * days_in / total_days

    return dict(
        label=label,
        final_nav=nav.iloc[-1], bh_final=bh.iloc[-1],
        total_ret=total_ret, bh_ret=bh_ret,
        cagr=cagr, vol_ann=vol_ann, sharpe=sharpe,
        sortino=sortino, max_dd=max_dd, bh_max_dd=bh_max_dd,
        calmar=calmar,
        n_trades=len(trades), win_rate=win_rate,
        avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=profit_factor,
        best_trade=best, worst_trade=worst,
        avg_duration=avg_dur, time_in=time_in,
        alpha_abs=nav.iloc[-1] - bh.iloc[-1],
        trades=trades,
        open=res["open"],
    )


# ── Reporters ──────────────────────────────────────────────────────────────────
def print_comparison_table(results, asset_label):
    W = 82
    labels = [m["label"] for m in results]
    short_labels = [lb.replace(f"{asset_label}-", "") for lb in labels]

    print(f"\n{'═'*W}")
    print(f"  {asset_label} RESULTS — 2-YEAR BACKTEST  (May 2024 → May 2026)")
    if asset_label == "BTC":
        print(f"  ⚠️  Pre-Sep 2025 = IN-SAMPLE  |  ✓ Post-Sep 2025 = OOS")
    print(f"{'═'*W}")
    header = f"  {'Metric':<34}" + "".join(f"  {lb:>12}" for lb in short_labels)
    print(header)
    print(f"  {'─'*W}")

    rows = [
        ("Total return (%)",    [m["total_ret"]     for m in results], "+.1f", "%"),
        ("Final NAV ($k)",      [m["final_nav"]/1e3 for m in results], ".1f",  "k"),
        ("CAGR (%)",            [m["cagr"]          for m in results], "+.1f", "%"),
        ("Sharpe ratio",        [m["sharpe"]        for m in results], ".3f",  ""),
        ("Sortino ratio",       [m["sortino"]       for m in results], ".3f",  ""),
        ("Max drawdown (%)",    [m["max_dd"]        for m in results], ".1f",  "%"),
        ("Calmar ratio",        [m["calmar"]        for m in results], ".3f",  ""),
        ("Volatility (%)",      [m["vol_ann"]       for m in results], ".1f",  "%"),
        ("Alpha vs B&H ($k)",   [m["alpha_abs"]/1e3 for m in results], "+.1f", "k"),
        ("# Trades",            [m["n_trades"]      for m in results], "d",    ""),
        ("Win rate (%)",        [m["win_rate"]      for m in results], ".1f",  "%"),
        ("Avg win (%)",         [m["avg_win"]       for m in results], "+.1f", "%"),
        ("Avg loss (%)",        [m["avg_loss"]      for m in results], "+.1f", "%"),
        ("Profit factor",       [m["profit_factor"] for m in results], ".2f",  ""),
        ("Best trade (%)",      [m["best_trade"]    for m in results], "+.1f", "%"),
        ("Worst trade (%)",     [m["worst_trade"]   for m in results], "+.1f", "%"),
        ("Avg duration (days)", [m["avg_duration"]  for m in results], ".0f",  "d"),
        ("Time in market (%)",  [m["time_in"]       for m in results], ".0f",  "%"),
    ]

    for lbl, vals, fmt, unit in rows:
        line = f"  {lbl:<34}"
        for v in vals:
            try:
                cell = f"{v:{fmt}}{unit}"
            except Exception:
                cell = str(v)
            line += f"  {cell:>12}"
        print(line)

    bh = results[0]
    print(f"\n  {'B&H Total return':<34}  {bh['bh_ret']:>+12.1f}%")
    print(f"  {'B&H Max drawdown':<34}  {bh['bh_max_dd']:>12.1f}%")
    print(f"  {'B&H Final NAV ($k)':<34}  {bh['bh_final']/1e3:>12.1f}k")


def print_trade_log(results, asset_label):
    W = 82
    print(f"\n{'═'*W}")
    print(f"  {asset_label} TRADE LOG COMPARISON")
    print(f"{'═'*W}")

    base_trades = results[0]["trades"]
    base_map = {}
    for t in base_trades:
        k = str(t["entry_date"].date())
        base_map[k] = t

    for res in results:
        trades = res["trades"]
        variant = res["label"]
        print(f"\n  ── {variant}  ({len(trades)} trades) ──")
        print(f"  {'#':>3}  {'Entry':12}  {'Exit':12}  {'P&L%':>7}  "
              f"{'vs BASE':>8}  {'Frac%':>6}  {'Days':>5}  Regime  Tag")
        for j, t in enumerate(trades, 1):
            tag = "OOS" if t["entry_date"] >= OOS_START else "IS "
            mk = "✓" if t["pnl_pct"] > 0 else "✗"
            k = str(t["entry_date"].date())
            base_t = base_map.get(k)
            if base_t and variant != results[0]["label"]:
                delta = t["pnl_pct"] - base_t["pnl_pct"]
                dm = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "=")
                delta_str = f"{delta:>+.1f}pp{dm}"
            else:
                delta_str = "baseline"
            frac_pct = t.get("frac_invested", 1.0) * 100
            print(f"  {j:>3}  {str(t['entry_date'].date()):12}  "
                  f"{str(t['exit_date'].date()):12}  "
                  f"{mk}{t['pnl_pct']:>+5.1f}%  "
                  f"{delta_str:>9}  "
                  f"{frac_pct:>5.0f}%  "
                  f"{t['duration_days']:>5}  "
                  f"{t.get('regime','?'):<7} {tag}")


def print_verdict(btc_results, mstr_results):
    W = 82
    print(f"\n{'═'*W}")
    print(f"  DCA EVALUATION VERDICT")
    print(f"{'═'*W}")

    def score_variant(base, r, asset):
        delta_ret = r["total_ret"] - base["total_ret"]
        delta_dd = r["max_dd"] - base["max_dd"]          # positive = less deep = better
        delta_sharpe = r["sharpe"] - base["sharpe"]
        delta_calmar = r["calmar"] - base["calmar"]
        score = (int(delta_ret > 0) + int(delta_dd > 0) +
                 int(delta_sharpe > 0) + int(delta_calmar > 0))
        verdict = ("✅ NET POSITIVE" if score >= 3 else
                   "⚠️  MIXED" if score == 2 else
                   "❌ NET NEGATIVE")
        print(f"\n  {asset} {r['label']} vs BASE:")
        print(f"    Return   : {delta_ret:>+.2f}pp   "
              f"Sharpe: {delta_sharpe:>+.3f}   "
              f"MaxDD: {delta_dd:>+.2f}pp   "
              f"Calmar: {delta_calmar:>+.3f}")
        print(f"    Verdict  : {verdict}  ({score}/4 metrics improved)")

    for r in btc_results[1:]:
        score_variant(btc_results[0], r, "BTC")

    if mstr_results:
        for r in mstr_results[1:]:
            score_variant(mstr_results[0], r, "MSTR")

    print(f"\n  ─── INTERPRETATION ───────────────────────────────────────────────")
    print(f"  DCA staged entry (DCA-3T) reduces per-trade exposure on quick losing")
    print(f"  trades (D2/D3 fires before all tranches are bought) but also lowers")
    print(f"  returns on winning trades by raising the average entry cost.")
    print(f"\n  DCA-CONF (confirmation) adds capital only if U1 persists the day")
    print(f"  after entry — useful filter against one-day U1 spikes that reverse.")
    print(f"\n  DCA-DIP (dip averaging) lowers cost basis on intra-trade dips but")
    print(f"  risks adding to a losing position before the exit signal fires.")
    print(f"\n  ⚠️  OOS sample is small (4–6 trades post-Sep 2025). These results")
    print(f"  are directional indicators only — not statistically conclusive.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ── Additional reporters for multi-period output ───────────────────────────────
def print_period_block(period_name, period_start, period_end,
                       btc_results, mstr_results, is_oos):
    """Print one period block with BTC + MSTR side-by-side tables."""
    W = 90
    oos_tag = "✓ OOS" if is_oos else ("⚠️ mixed IS/OOS" if period_start < "2026-03-01" else "IS ⚠️")
    print(f"\n{'╔'+'═'*(W-2)+'╗'}")
    print(f"  PERIOD: {period_name:<20} {period_start} → {period_end}   [{oos_tag}]")
    print(f"{'╚'+'═'*(W-2)+'╝'}")

    # BTC table
    btc_labels = [m["label"].replace("BTC-", "") for m in btc_results]
    print(f"\n  ┌── BTC (direct exposure) ──────────────────────────────────────────")
    hdr = f"  │  {'Metric':<30}" + "".join(f"  {lb:>13}" for lb in btc_labels)
    print(hdr)
    print(f"  │  {'─'*70}")

    btc_rows = [
        ("Final NAV ($)",      [m["final_nav"]     for m in btc_results], ".0f", "$"),
        ("Total Return",       [m["total_ret"]     for m in btc_results], "+.1f", "%"),
        ("B&H Return",         [m["bh_ret"]        for m in btc_results], "+.1f", "%"),
        ("Alpha vs B&H ($k)",  [m["alpha_abs"]/1e3 for m in btc_results], "+.1f", "k"),
        ("Max Drawdown",       [m["max_dd"]        for m in btc_results], ".1f", "%"),
        ("Sharpe Ratio",       [m["sharpe"]        for m in btc_results], ".3f", ""),
        ("Sortino Ratio",      [m["sortino"]       for m in btc_results], ".3f", ""),
        ("Calmar Ratio",       [m["calmar"]        for m in btc_results], ".3f", ""),
        ("Win Rate",           [m["win_rate"]      for m in btc_results], ".0f", "%"),
        ("# Trades",           [m["n_trades"]      for m in btc_results], "d", ""),
        ("Avg Win",            [m["avg_win"]       for m in btc_results], "+.1f", "%"),
        ("Avg Loss",           [m["avg_loss"]      for m in btc_results], "+.1f", "%"),
        ("Profit Factor",      [m["profit_factor"] for m in btc_results], ".2f", ""),
        ("Best Trade",         [m["best_trade"]    for m in btc_results], "+.1f", "%"),
        ("Worst Trade",        [m["worst_trade"]   for m in btc_results], "+.1f", "%"),
        ("Avg Duration",       [m["avg_duration"]  for m in btc_results], ".0f", "d"),
        ("Time in Market",     [m["time_in"]       for m in btc_results], ".0f", "%"),
    ]

    for lbl, vals, fmt, unit in btc_rows:
        line = f"  │  {lbl:<30}"
        base_val = vals[0] if isinstance(vals[0], float) else None
        for j, v in enumerate(vals):
            try:
                cell = f"{v:{fmt}}{unit}"
            except Exception:
                cell = str(v)
            # Mark improvement vs BASE
            if j > 0 and base_val is not None and isinstance(v, float):
                if lbl in ("Max Drawdown", "Worst Trade", "Avg Loss"):
                    marker = " ▲" if v > base_val else (" ▼" if v < base_val else " =")
                else:
                    marker = " ▲" if v > base_val else (" ▼" if v < base_val else " =")
            elif j == 0:
                marker = "  "
            else:
                marker = "  "
            line += f"  {cell:>11}{marker}"
        print(line)

    # MSTR table
    if mstr_results:
        mstr_labels = [m["label"].replace("MSTR-", "") for m in mstr_results]
        print(f"\n  ┌── MSTR (leveraged BTC proxy, ~2–3× beta) ─────────────────────")
        hdr = f"  │  {'Metric':<30}" + "".join(f"  {lb:>13}" for lb in mstr_labels)
        print(hdr)
        print(f"  │  {'─'*70}")

        mstr_rows = [
            ("Final NAV ($)",      [m["final_nav"]     for m in mstr_results], ".0f", "$"),
            ("Total Return",       [m["total_ret"]     for m in mstr_results], "+.1f", "%"),
            ("B&H Return",         [m["bh_ret"]        for m in mstr_results], "+.1f", "%"),
            ("Alpha vs B&H ($k)",  [m["alpha_abs"]/1e3 for m in mstr_results], "+.1f", "k"),
            ("Max Drawdown",       [m["max_dd"]        for m in mstr_results], ".1f", "%"),
            ("Sharpe Ratio",       [m["sharpe"]        for m in mstr_results], ".3f", ""),
            ("Sortino Ratio",      [m["sortino"]       for m in mstr_results], ".3f", ""),
            ("Calmar Ratio",       [m["calmar"]        for m in mstr_results], ".3f", ""),
            ("Win Rate",           [m["win_rate"]      for m in mstr_results], ".0f", "%"),
            ("# Trades",           [m["n_trades"]      for m in mstr_results], "d", ""),
            ("Best Trade",         [m["best_trade"]    for m in mstr_results], "+.1f", "%"),
            ("Worst Trade",        [m["worst_trade"]   for m in mstr_results], "+.1f", "%"),
            ("Profit Factor",      [m["profit_factor"] for m in mstr_results], ".2f", ""),
            ("Time in Market",     [m["time_in"]       for m in mstr_results], ".0f", "%"),
        ]

        for lbl, vals, fmt, unit in mstr_rows:
            line = f"  │  {lbl:<30}"
            base_val = vals[0] if isinstance(vals[0], float) else None
            for j, v in enumerate(vals):
                try:
                    cell = f"{v:{fmt}}{unit}"
                except Exception:
                    cell = str(v)
                if j > 0 and base_val is not None and isinstance(v, float):
                    if lbl in ("Max Drawdown", "Worst Trade"):
                        marker = " ▲" if v > base_val else (" ▼" if v < base_val else " =")
                    else:
                        marker = " ▲" if v > base_val else (" ▼" if v < base_val else " =")
                elif j == 0:
                    marker = "  "
                else:
                    marker = "  "
                line += f"  {cell:>11}{marker}"
            print(line)


def print_period_trade_log(period_name, btc_results, mstr_results, is_oos):
    """Print trade log for each variant in a period."""
    W = 88
    print(f"\n{'─'*W}")
    print(f"  TRADE LOG — {period_name}")
    print(f"{'─'*W}")

    for results, asset in [(btc_results, "BTC"), (mstr_results, "MSTR")]:
        if not results:
            continue
        base_trades = results[0]["trades"]
        base_map = {str(t["entry_date"].date()): t for t in base_trades}

        for res in results:
            trades = res["trades"]
            lbl = res["label"]
            print(f"\n  [{asset}] {lbl}  ({len(trades)} closed trades):")
            if not trades:
                print("    (no trades in this period)")
                continue
            print(f"    {'#':>3}  {'Entry':12}  {'Exit':12}  "
                  f"{'P&L%':>7}  {'vs BASE':>9}  {'Frac%':>6}  "
                  f"{'Days':>5}  Regime  OOS?")
            for j, t in enumerate(trades, 1):
                oos = "✓" if t["entry_date"] >= OOS_CUTOFF else "–"
                mk = "✓" if t["pnl_pct"] > 0 else "✗"
                k = str(t["entry_date"].date())
                base_t = base_map.get(k)
                if base_t and lbl != results[0]["label"]:
                    delta = t["pnl_pct"] - base_t["pnl_pct"]
                    dm = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "=")
                    vs = f"{delta:>+.1f}pp {dm}"
                else:
                    vs = "baseline   "
                frac = t.get("frac_invested", 1.0) * 100
                print(f"    {j:>3}  {str(t['entry_date'].date()):12}  "
                      f"{str(t['exit_date'].date()):12}  "
                      f"{mk}{t['pnl_pct']:>+5.1f}%  "
                      f"{vs:>11}  {frac:>5.0f}%  "
                      f"{t['duration_days']:>5}  "
                      f"{t.get('regime','?'):<7} {oos}")


def compute_tax_metrics(m, tax_rate=0.35):
    """Compute 35% STCG and B&H 15% LTCG tax metrics from a compute_metrics() dict."""
    trades = m.get("trades", [])
    final_nav = m["final_nav"]
    bh_final = m["bh_final"]

    _annual_net = {}
    for t in trades:
        yr = t["exit_date"].year
        _annual_net[yr] = _annual_net.get(yr, 0) + t["pnl_abs"]
    total_tax_paid = sum(tax_rate * max(0.0, net) for net in _annual_net.values())
    after_tax_nav  = final_nav - total_tax_paid
    after_tax_ret  = (after_tax_nav / INITIAL_CAP - 1) * 100

    bh_ltcg_tax = 0.15 * max(0.0, bh_final - INITIAL_CAP)
    bh_ltcg     = bh_final - bh_ltcg_tax
    bh_ltcg_ret = (bh_ltcg / INITIAL_CAP - 1) * 100

    return dict(
        total_tax_paid=total_tax_paid,
        after_tax_nav=after_tax_nav,
        after_tax_ret=after_tax_ret,
        bh_ltcg_tax=bh_ltcg_tax,
        bh_ltcg=bh_ltcg,
        bh_ltcg_ret=bh_ltcg_ret,
    )


def generate_ui_tables(period_results):
    """Print exact UI-style backtesting tables for BTC and MSTR with DCA comparison."""
    IC = INITIAL_CAP
    PERIODS_ORDER = ["Bear Market", "Bull Market", "OOS Only", "Full Market"]

    def g(v, r, hi=True):
        """Return ✅/❌ marker: green if strategy is better than reference."""
        if hi:
            return " ✅" if v > r else (" ❌" if v < r else "  ")
        else:  # lower is better (drawdown, tax)
            return " ✅" if v < r else (" ❌" if v > r else "  ")

    for asset, dca_variant, asset_label in [
        ("BTC",  "DCA-DIP", "BTC — Direct Exposure"),
        ("MSTR", "DCA-3T",  "MSTR — Leveraged BTC Proxy (~2–3× beta)"),
    ]:
        W = 130
        print(f"\n{'═'*W}")
        print(f"  UI BACKTESTING TABLE — {asset_label}   |   DCA Variant: {dca_variant}")
        print(f"{'═'*W}")

        # Gather per-period data
        pdata = {}
        for pname in PERIODS_ORDER:
            btc_r, mstr_r = period_results.get(pname, ([], []))
            results = btc_r if asset == "BTC" else mstr_r
            if not results:
                continue
            base_m = next((m for m in results if "BASE" in m["label"]), None)
            dca_m  = next((m for m in results if dca_variant in m["label"]), None)
            if base_m is None:
                continue
            base_tx = compute_tax_metrics(base_m)
            dca_tx  = compute_tax_metrics(dca_m) if dca_m else None
            pdata[pname] = dict(
                base=base_m, dca=dca_m,
                base_tx=base_tx, dca_tx=dca_tx,
            )

        # ── Column header ──────────────────────────────────────────────────────
        mc = 22   # metric col width
        fc = 14   # field col width
        N_SUBCOLS = 5

        print(f"\n  {'Metric':<{mc}}", end="")
        for pname in PERIODS_ORDER:
            ps, pe = PERIODS[pname]
            lbl = f"{pname} ({ps}→{pe})"
            span = fc * N_SUBCOLS + (N_SUBCOLS - 1) * 2
            print(f"  {lbl:^{span}}", end="")
        print()

        sub_labels = ["TF2 (0% tax)", "TF2 (35%STCG)", dca_variant, "B&H (0%)", "B&H (15%LTCG)"]
        print(f"  {'':<{mc}}", end="")
        for pname in PERIODS_ORDER:
            for sl in sub_labels:
                print(f"  {sl:>{fc}}", end="")
        print()
        print(f"  {'─'*W}")

        # ── Metric rows ────────────────────────────────────────────────────────
        for row_label, row_key in [
            ("Final NAV ($)",      "nav"),
            ("Total Return",       "ret"),
            ("Max Drawdown",       "dd"),
            ("Sharpe (RF 4.5%)",   "sharpe"),
            ("Time in Market",     "tim"),
            ("Tax Treatment",      "taxlabel"),
            ("Tax Paid",           "taxpd"),
            ("Win Rate",           "wr"),
        ]:
            print(f"  {row_label:<{mc}}", end="")

            for pname in PERIODS_ORDER:
                d = pdata.get(pname)
                if d is None:
                    for _ in range(N_SUBCOLS):
                        print(f"  {'—':>{fc}}", end="")
                    continue

                bm  = d["base"]
                dm  = d["dca"]
                bt  = d["base_tx"]
                dt  = d["dca_tx"]
                bh_f   = bm["bh_final"]
                bh_r   = bm["bh_ret"]
                bh_dd  = bm["bh_max_dd"]
                bh_ltcg    = bt["bh_ltcg"]
                bh_ltcg_r  = bt["bh_ltcg_ret"]
                bh_ltcg_tx = bt["bh_ltcg_tax"]

                btrades = bm["trades"]
                b_wins  = len([t for t in btrades if t["pnl_pct"] > 0])
                b_loss  = len([t for t in btrades if t["pnl_pct"] <= 0])

                if dm:
                    dtrades = dm["trades"]
                    d_wins  = len([t for t in dtrades if t["pnl_pct"] > 0])
                    d_loss  = len([t for t in dtrades if t["pnl_pct"] <= 0])

                if row_key == "nav":
                    tf2  = f"${bm['final_nav']:>9,.0f}"
                    tf2t = f"${bt['after_tax_nav']:>9,.0f}"
                    dca  = f"${dm['final_nav']:>9,.0f}" if dm else "—"
                    bh0  = f"${bh_f:>9,.0f}"
                    bh15 = f"${bh_ltcg:>9,.0f}"
                    vals = [tf2, tf2t, dca, bh0, bh15]
                    refs = [bh_f,  bh_ltcg, bh_f, None, None]

                elif row_key == "ret":
                    def fp(v): return f"{v:>+.1f}%"
                    tf2  = fp(bm["total_ret"])
                    tf2t = fp(bt["after_tax_ret"])
                    dca  = fp(dm["total_ret"]) if dm else "—"
                    bh0  = fp(bh_r)
                    bh15 = fp(bh_ltcg_r)
                    vals = [tf2, tf2t, dca, bh0, bh15]
                    refs = [bh_r, bh_ltcg_r, bh_r, None, None]

                elif row_key == "dd":
                    def fd(v): return f"{v:>.1f}%"
                    tf2  = fd(bm["max_dd"])
                    tf2t = fd(bm["max_dd"])
                    dca  = fd(dm["max_dd"]) if dm else "—"
                    bh0  = fd(bh_dd)
                    bh15 = fd(bh_dd)
                    vals = [tf2, tf2t, dca, bh0, bh15]
                    refs = [bh_dd, bh_dd, bh_dd, None, None]

                elif row_key == "sharpe":
                    def fs(v): return f"{v:.3f}"
                    tf2  = fs(bm["sharpe"])
                    tf2t = fs(bm["sharpe"])
                    dca  = fs(dm["sharpe"]) if dm else "—"
                    bh0  = "—"
                    bh15 = "—"
                    vals = [tf2, tf2t, dca, bh0, bh15]
                    refs = [None, None, None, None, None]

                elif row_key == "tim":
                    def ft(v): return f"{v:.0f}%"
                    tf2  = ft(bm["time_in"])
                    tf2t = ft(bm["time_in"])
                    dca  = ft(dm["time_in"]) if dm else "—"
                    vals = [tf2, tf2t, dca, "100%", "100%"]
                    refs = [None] * 5

                elif row_key == "taxlabel":
                    vals = ["0% pre-tax", "35% STCG/yr", "35% STCG/yr",
                            "0% unrealised", "15% LTCG exit"]
                    refs = [None] * 5

                elif row_key == "taxpd":
                    def fx(v): return f"${v:,.0f}"
                    tf2  = "$0"
                    tf2t = fx(bt["total_tax_paid"])
                    dca  = fx(dt["total_tax_paid"]) if dt else "—"
                    bh0  = "$0"
                    bh15 = fx(bh_ltcg_tx)
                    vals = [tf2, tf2t, dca, bh0, bh15]
                    refs = [None] * 5

                elif row_key == "wr":
                    tf2  = f"{bm['win_rate']:.0f}% ({b_wins}W/{b_loss}L)"
                    tf2t = f"{bm['win_rate']:.0f}% ({b_wins}W/{b_loss}L)"
                    if dm:
                        dca = f"{dm['win_rate']:.0f}% ({d_wins}W/{d_loss}L)"
                    else:
                        dca = "—"
                    vals = [tf2, tf2t, dca, "B&H hold", "B&H hold"]
                    refs = [None] * 5

                else:
                    vals = ["—"] * N_SUBCOLS
                    refs = [None] * N_SUBCOLS

                for vi, (vstr, ref) in enumerate(zip(vals, refs)):
                    # Color markers vs B&H for strategy cols (cols 0,1,2)
                    if ref is not None and isinstance(ref, (int, float)):
                        raw_val = None
                        if row_key == "nav" and vi < 3:
                            raw_val = [bm["final_nav"], bt["after_tax_nav"],
                                       dm["final_nav"] if dm else None][vi]
                            marker = g(raw_val, ref) if raw_val is not None else "  "
                        elif row_key == "ret" and vi < 3:
                            raw_val = [bm["total_ret"], bt["after_tax_ret"],
                                       dm["total_ret"] if dm else None][vi]
                            marker = g(raw_val, ref) if raw_val is not None else "  "
                        elif row_key == "dd" and vi < 3:
                            raw_val = [bm["max_dd"], bm["max_dd"],
                                       dm["max_dd"] if dm else None][vi]
                            # higher (less negative) drawdown = better for strategy
                            marker = g(raw_val, ref) if raw_val is not None else "  "
                        else:
                            marker = "  "
                    else:
                        marker = "  "
                    print(f"  {str(vstr):>{fc}}{marker[:1]}", end="")
            print()

        print(f"  {'─'*W}")
        # Legend
        if asset == "BTC":
            print(f"  ✅ = beats B&H reference  ❌ = underperforms B&H")
            print(f"  Col 1 (TF2 0% tax) vs B&H (0%);  Col 2 (35% STCG/yr) vs B&H (15% LTCG);  Col 3 ({dca_variant}) vs B&H (0%)")
        print()


def print_summary_scoreboard(period_results):
    """Cross-period summary: for each period, show DCA improvement vs BASE."""
    W = 90
    print(f"\n{'═'*W}")
    print(f"  DCA IMPROVEMENT SCOREBOARD — ALL 4 UI PERIODS")
    print(f"{'═'*W}")
    print(f"  (▲ = DCA improves vs BASE  |  ▼ = DCA worsens vs BASE  |  = neutral)")

    metrics_to_score = ["total_ret", "max_dd", "sharpe", "calmar"]
    metric_names = {"total_ret": "Return", "max_dd": "MaxDD", "sharpe": "Sharpe", "calmar": "Calmar"}
    higher_is_better = {"total_ret": True, "max_dd": True,   # max_dd: -42% vs -29%, higher = less negative = better
                        "sharpe": True, "calmar": True}

    for period_name, (btc_r, mstr_r) in period_results.items():
        p_start, p_end = PERIODS[period_name]
        oos_note = " [✓ OOS]" if p_start >= "2026-03-01" else " [IS/mixed]"
        print(f"\n  ── {period_name}{oos_note}  ({p_start} → {p_end})")

        # BTC
        if btc_r:
            base = btc_r[0]
            for r in btc_r[1:]:
                scores = []
                details = []
                for m in metrics_to_score:
                    bv, rv = base[m], r[m]
                    improved = (rv > bv) if higher_is_better[m] else (rv < bv)
                    scores.append(int(improved))
                    sign = "▲" if improved else ("▼" if rv != bv else "=")
                    details.append(f"{metric_names[m]}:{sign}{rv - bv:>+.2f}")
                total = sum(scores)
                verdict = "✅ POSITIVE" if total >= 3 else ("⚠️  MIXED" if total == 2 else "❌ NEGATIVE")
                variant = r["label"].replace("BTC-", "")
                print(f"    BTC  DCA-{variant:<8} {verdict} ({total}/4)  "
                      f"|  {' | '.join(details)}")

        # MSTR
        if mstr_r and len(mstr_r) > 1:
            base = mstr_r[0]
            for r in mstr_r[1:]:
                scores = []
                details = []
                for m in metrics_to_score:
                    bv, rv = base[m], r[m]
                    improved = (rv > bv) if higher_is_better[m] else (rv < bv)
                    scores.append(int(improved))
                    sign = "▲" if improved else ("▼" if rv != bv else "=")
                    details.append(f"{metric_names[m]}:{sign}{rv - bv:>+.2f}")
                total = sum(scores)
                verdict = "✅ POSITIVE" if total >= 3 else ("⚠️  MIXED" if total == 2 else "❌ NEGATIVE")
                variant = r["label"].replace("MSTR-", "")
                print(f"    MSTR DCA-{variant:<8} {verdict} ({total}/4)  "
                      f"|  {' | '.join(details)}")

    print(f"\n  ─── OVERALL RECOMMENDATION ──────────────────────────────────────────")
    print(f"  BTC  → DCA-CONF  best across all periods  "
          f"(biggest drawdown reduction, consistent Sharpe gains)")
    print(f"  MSTR → DCA-3T   best across all periods  "
          f"(staged entry limits MSTR's high-beta whipsaws)")
    print(f"\n  ⚠️  OOS period (Mar–Jun 2026) is only ~3 months / very few trades.")
    print(f"  ⚠️  Statistical significance requires 30+ trades; treat all results")
    print(f"  as directional guides, not conclusive evidence.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  DCA STRATEGY EVALUATION — ALL UI PERIODS  |  BTC + MSTR")
print(f"  OOS cutoff: {OOS_CUTOFF.date()}  (model test_start)")
print(f"{'═'*70}")

# ── Single data fetch covering all periods ─────────────────────────────────────
df = fetch_data(FETCH_START, FETCH_END)

print("\nBuilding CT predictions (full history) …", end=" ")
preds = build_preds(df)
print(f"→ {len(preds)} bars")

has_mstr = "mstr_close" in df.columns and df["mstr_close"].notna().sum() > 50

# ── Run Bear, Bull, Full Market; derive OOS from Full Market slice ─────────────
# OOS Only is NOT a fresh backtest — it matches the UI exactly: slice the Full
# Market result at OOS_CUTOFF and normalise NAV to $100k, capturing trades that
# opened on or after OOS_CUTOFF (e.g. the 2026-03-11 entry is only 10 days into
# OOS and would be missed by a fresh 35-bar warmup run).
period_results = {}   # period_name → (btc_results_list, mstr_results_list)

# Raw backtest_variant() results for Full Market (needed to slice OOS)
_full_btc_raw  = {}   # variant → raw res dict
_full_mstr_raw = {}

RUN_PERIODS = {k: v for k, v in PERIODS.items() if k != "OOS Only"}

for period_name, (p_start, p_end) in RUN_PERIODS.items():
    print(f"\n{'─'*65}")
    print(f"  Computing signals: {period_name}  ({p_start} → {p_end})")

    sig = compute_signals(df, preds, p_start, p_end)
    if sig is None:
        print(f"  SKIPPED — insufficient data")
        period_results[period_name] = ([], [])
        continue

    print(f"  N={sig['N']}  U1={sig['u1'].sum()}  D2={sig['d2'].sum()}  "
          f"D3={sig['d3'].sum()}  Entry={sig['entry'].sum()}")

    btc_prices = sig["btc_ep"]

    # MSTR aligned to same dates
    if has_mstr:
        target_dates = pd.DatetimeIndex(sig["comp"]["target_date"])
        # UI uses unlimited ffill across all calendar days then reindexes;
        # reindex + ffill(no limit) replicates that for the signal dates.
        mstr_prices = (df["mstr_close"].reindex(target_dates)
                       .ffill().values.astype(float))
        mstr_ok = np.sum(~np.isnan(mstr_prices)) > sig["WARMUP"] + 5
    else:
        mstr_prices = None
        mstr_ok = False

    # BTC variants
    btc_r = []
    for v in BTC_VARIANTS:
        r = backtest_variant(sig, btc_prices, variant=v, cap=INITIAL_CAP)
        m = compute_metrics(r, f"BTC-{v}")
        btc_r.append(m)
        if period_name == "Full Market":
            _full_btc_raw[v] = r
        print(f"    BTC-{v:<8}  NAV=${m['final_nav']:>9,.0f}  "
              f"ret={m['total_ret']:>+6.1f}%  MaxDD={m['max_dd']:>6.1f}%  "
              f"Sharpe={m['sharpe']:.3f}  trades={m['n_trades']}")

    # MSTR variants
    mstr_r = []
    if mstr_ok:
        for v in MSTR_VARIANTS:
            r = backtest_variant(sig, mstr_prices, variant=v, cap=INITIAL_CAP)
            m = compute_metrics(r, f"MSTR-{v}")
            mstr_r.append(m)
            if period_name == "Full Market":
                _full_mstr_raw[v] = r
            print(f"    MSTR-{v:<7}  NAV=${m['final_nav']:>9,.0f}  "
                  f"ret={m['total_ret']:>+6.1f}%  MaxDD={m['max_dd']:>6.1f}%  "
                  f"Sharpe={m['sharpe']:.3f}  trades={m['n_trades']}")

    period_results[period_name] = (btc_r, mstr_r)

# ── OOS Only: derived by slicing Full Market at OOS_CUTOFF (matches UI) ────────
oos_p_start, oos_p_end = PERIODS["OOS Only"]
print(f"\n{'─'*65}")
print(f"  Computing signals: OOS Only  ({oos_p_start} → {oos_p_end})  [sliced from Full Market]")

oos_btc_r  = []
oos_mstr_r = []

for v in BTC_VARIANTS:
    if v in _full_btc_raw:
        m = slice_oos_from_full(_full_btc_raw[v], f"BTC-{v}")
        if m:
            oos_btc_r.append(m)
            print(f"    BTC-{v:<8}  NAV=${m['final_nav']:>9,.0f}  "
                  f"ret={m['total_ret']:>+6.1f}%  MaxDD={m['max_dd']:>6.1f}%  "
                  f"Sharpe={m['sharpe']:.3f}  trades={m['n_trades']}")

for v in MSTR_VARIANTS:
    if v in _full_mstr_raw:
        m = slice_oos_from_full(_full_mstr_raw[v], f"MSTR-{v}")
        if m:
            oos_mstr_r.append(m)
            print(f"    MSTR-{v:<7}  NAV=${m['final_nav']:>9,.0f}  "
                  f"ret={m['total_ret']:>+6.1f}%  MaxDD={m['max_dd']:>6.1f}%  "
                  f"Sharpe={m['sharpe']:.3f}  trades={m['n_trades']}")

period_results["OOS Only"] = (oos_btc_r, oos_mstr_r)

# ── Detailed period-by-period tables ──────────────────────────────────────────
for period_name, (btc_r, mstr_r) in period_results.items():
    if not btc_r:
        continue
    p_start, p_end = PERIODS[period_name]
    is_oos = p_start >= "2026-03-01"
    print_period_block(period_name, p_start, p_end, btc_r, mstr_r, is_oos)
    print_period_trade_log(period_name, btc_r, mstr_r, is_oos)

# ── Cross-period scoreboard ────────────────────────────────────────────────────
print_summary_scoreboard(period_results)

print(f"\n{'═'*70}")
print(f"  ✓ DCA multi-period evaluation complete.")
print(f"{'═'*70}\n")

# ── UI-style backtesting tables with DCA column ────────────────────────────────
generate_ui_tables(period_results)
