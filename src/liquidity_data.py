"""
Net global liquidity series for the BTC daily high/low model.

The construct traders call "net liquidity" is the Fed's balance sheet minus the
two big sterilising liabilities — cash parked at the Fed by the Treasury and by
money funds:

    net_liq = WALCL - WTREGEN - RRPONTSYD

    WALCL     Fed total assets                       weekly (Wed), $mn
    WTREGEN   Treasury General Account, week average weekly (Wed), $mn
    RRPONTSYD Overnight reverse repo                 daily,        $bn

"Global" liquidity adds the other major central banks. Their balance sheets are
monthly and land on FRED weeks-to-months late (ECBASSETSW is the only weekly
one), so a same-day-honest global aggregate is mostly the Fed plus a stale
constant. This module therefore builds the Fed net-liquidity core and adds the
ECB's weekly balance sheet in USD as an optional extension.

RELEASE TIMING is the whole ballgame here. Every series is stamped with its
*observation* date, but H.4.1 for the week ending Wednesday is not published
until Thursday 16:30 ET, and the ON-RRP print for day D lands ~13:15 ET on D
itself — both after the 12:00-UTC bar the model scores. So an observation is
mapped to the first 12:00-UTC bar that could actually have seen it, and a
further ``safety_lag_days`` is applied on top. Getting this wrong manufactures
a look-ahead edge that evaporates in live trading.
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# series id -> (unit multiplier to $mn, observation->availability lag in days)
#   weekly H.4.1 series (WALCL, WTREGEN, ECBASSETSW): obs dated Wed, released
#     Thu 16:30 ET, so the Friday 12:00-UTC bar is the first that can use it.
#   RRPONTSYD: obs dated D, released ~13:15 ET on D, after the 12:00-UTC bar,
#     so bar D+1 is the first that can use it.
SERIES = {
    "WALCL":     (1.0,    2),
    "WTREGEN":   (1.0,    2),
    "RRPONTSYD": (1000.0, 1),
    "ECBASSETSW": (1.0,   4),   # EUR mn, weekly, published with a longer lag
    "DEXUSEU":   (1.0,    1),   # USD per EUR, daily
}


def _fetch_fred(sid: str, cache_dir: Path | None = None) -> pd.Series:
    """Download one FRED series as a date-indexed float Series (NaNs dropped)."""
    cache = (cache_dir / f"{sid}.csv") if cache_dir else None
    if cache is not None and cache.exists():
        raw = cache.read_text()
    else:
        req = urllib.request.Request(FRED_CSV.format(sid=sid),
                                     headers={"User-Agent": "research/1.0"})
        raw = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read().decode()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw)
    df = pd.read_csv(io.StringIO(raw))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = df[date_col]
    return s.dropna()


def load_net_liquidity(start: str = "2018-01-01", end: str | None = None,
                       cache_dir: Path | None = None,
                       safety_lag_days: int = 1,
                       include_ecb: bool = True) -> pd.DataFrame:
    """Daily, point-in-time-honest net liquidity levels in $mn.

    Returns columns: net_liq, walcl, tga, rrp (and ecb_usd / global_liq when
    ``include_ecb``). Every column is already lagged to what was *knowable* at
    a 12:00-UTC bar on that date, so callers must not shift again.
    """
    wanted = ["WALCL", "WTREGEN", "RRPONTSYD"] + (["ECBASSETSW", "DEXUSEU"]
                                                  if include_ecb else [])
    idx = pd.date_range(start, end or pd.Timestamp.utcnow().tz_localize(None).normalize(), freq="D")
    out = {}
    for sid in wanted:
        mult, lag = SERIES[sid]
        s = _fetch_fred(sid, cache_dir) * mult
        # move each observation to the first date it could be acted on, then
        # hold it flat until the next release
        s.index = s.index + pd.Timedelta(days=lag + safety_lag_days)
        out[sid] = s[~s.index.duplicated(keep="last")].reindex(idx).ffill()

    df = pd.DataFrame(out, index=idx)
    df = df.rename(columns={"WALCL": "walcl", "WTREGEN": "tga",
                            "RRPONTSYD": "rrp"})
    df["net_liq"] = df["walcl"] - df["tga"] - df["rrp"]

    if include_ecb:
        # ECBASSETSW is EUR mn; DEXUSEU is USD per EUR
        df["ecb_usd"] = df["ECBASSETSW"] * df["DEXUSEU"]
        df["global_liq"] = df["net_liq"] + df["ecb_usd"]
        df = df.drop(columns=["ECBASSETSW", "DEXUSEU"])

    return df.dropna(subset=["net_liq"])


def liquidity_features(liq: pd.DataFrame, index: pd.DatetimeIndex,
                       prefix: str = "liq") -> pd.DataFrame:
    """Stationary features from the net-liquidity level.

    Levels are trending and non-stationary, so nothing here uses a raw level.
    The blocks are: rate of change over 1w/4w/13w, position within a 1-year
    range, an impulse/acceleration term, the volatility of liquidity flows
    (the channel most plausibly linked to a *range* target), and the two
    fast-moving components (TGA, RRP) that traders actually watch.
    """
    # Compute on the liquidity panel's own full daily history and only then
    # reindex onto the model's bars — doing it the other way round would throw
    # away a year of rows to the 364-day rolling windows.
    l = liq.reindex(liq.index.union(index)).ffill()
    f = pd.DataFrame(index=l.index)

    net = l["net_liq"]
    ln = np.log(net.where(net > 0))
    for k in (7, 28, 91):
        f[f"{prefix}_d{k}"] = ln.diff(k)
    f[f"{prefix}_z364"] = ((net - net.rolling(364).mean())
                           / net.rolling(364).std())
    # impulse: is the 1-week pace running above the 4-week pace
    f[f"{prefix}_accel"] = ln.diff(7) - ln.diff(28) / 4.0
    # flow volatility — liquidity turbulence rather than direction
    f[f"{prefix}_vol28"] = ln.diff().rolling(28).std()

    # fast components, signed so that "+" = liquidity added to the system
    f[f"{prefix}_tga_d7"] = -np.log(l["tga"].where(l["tga"] > 0)).diff(7)
    f[f"{prefix}_rrp_d7"] = -(l["rrp"].diff(7) / net)
    f[f"{prefix}_walcl_d28"] = np.log(l["walcl"].where(l["walcl"] > 0)).diff(28)

    if "global_liq" in l.columns:
        g = l["global_liq"]
        lg = np.log(g.where(g > 0))
        f[f"{prefix}_g_d28"] = lg.diff(28)
        f[f"{prefix}_g_d91"] = lg.diff(91)
        f[f"{prefix}_g_z364"] = (g - g.rolling(364).mean()) / g.rolling(364).std()

    return f.replace([np.inf, -np.inf], np.nan).reindex(index)
