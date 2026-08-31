"""Binance hourly klines → 12:00-UTC daily bars, with hourly-gap healing.

Shared by the once-daily dataset pull (``scripts/pull_backtest_data.py``) and
the live Streamlit app (``app/btc_hourly_app.py``).  Both build the model's
12:00-UTC (7am-CT) daily bars out of Binance hourly klines and both drop a bar
whose 24 hours are not all present, so both are exposed to the same defect —
and when the healing lived only in the puller, the app kept showing the daily
signal bar a day behind while the committed dataset was healthy (observed
2026-08-31, twice: once for the dataset, once for the app).  One implementation
here, imported by both, is what keeps them in step.

The hosts below are NOT interchangeable.  ``api.binance.us`` is a SEPARATE
venue carrying ~0.3% of Binance.com's BTC volume; ``data-api.binance.vision``
mirrors Binance.com; ``api.binance.com`` itself answers HTTP 451 from
US-hosted infrastructure (GitHub Actions, Streamlit Community Cloud).  The
FIRST host that answers owns the whole series: the committed vintage's volume
history is on that venue's scale, the live app's rows are spliced onto that
same vintage by ``_seed_daily_raw_from_versioned``, and the vintage freeze pins
it — so a mid-series host swap would put two volume scales in one column.  Only
GAPS are borrowed from another host, rescaled (``heal_hourly_gaps``); changing
the primary venue outright is a deliberate re-baseline (``PULL_UNFROZEN=1``,
see DATA_CONSISTENCY.md).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

BINANCE_HOSTS = ("https://api.binance.us", "https://data-api.binance.vision",
                 "https://api.binance.com")
ANCHOR_HOUR_UTC = 12  # 7am CDT / 6am CST — the daily model's bar boundary

HOUR_MS = 3_600_000
DAY_MS  = 86_400_000

# ── hourly-gap healing ──────────────────────────────────────────────────────
# `api.binance.us` periodically serves no hourly kline at all for a run of
# hours — observed 2026-08-31: 04:00–12:00 UTC absent from its 1h *and* 1m
# series while the .vision mirror had every one of them.  A 12:00-UTC daily bar
# is only emitted when all 24 of its hours are present, so one such hole
# silently DROPS an otherwise-complete daily bar: the committed dataset stops a
# bar short, the BTC/MSTR/MSTU/ETH sleeve freezes on the previous bar, the
# publisher's freshness audit flags BTC stale and the day's target book is
# withheld — and in the live app every DAILY view (H/L forecast, CT signal,
# position) is computed from the older bar.
#
# Healing rule: for a bar whose 24-hour window has already CLOSED but whose
# hours are incomplete, take the missing hours from the next host that has
# them, with the donor's volume rescaled onto the primary venue's own scale
# (calibrated on the hours both hosts serve).  Closes agree across venues to
# ~0.01%, so prices splice cleanly; raw volumes do NOT (a ~300x step), and the
# model's volume features (`vol_chg_1` = log-diff, `vol_z_20`, `vol_ma_ratio`)
# would read an unscaled venue switch as a volume shock lasting the whole
# 20-bar window.  A donor whose prices disagree, or whose volume scale cannot
# be calibrated, is REFUSED — the bar is then dropped exactly as before, the
# audit correctly reports BTC stale and the app shows its "bar(s) behind"
# warning, rather than either carrying a spliced bar.
HEAL_LOOKBACK_DAYS      = 5     # the freeze-refreshable tail only (see FREEZE_TAIL)
HEAL_MIN_PRIMARY_HOURS  = 12    # patch a hole in a bar the venue traded; never import a whole bar
HEAL_MIN_OVERLAP        = 48    # hours both hosts must share to calibrate the volume scale
HEAL_MAX_PX_DIVERGENCE  = 0.01  # 1% median |close ratio − 1| tolerated between venues

_OHLCV = ["open", "high", "low", "close", "volume"]


def klines_page(host: str, symbol: str, start_ms: int, limit: int = 1000,
                timeout: int = 30):
    """One page of hourly klines from ONE host, or ``None`` if it does not
    answer 200 with a JSON list."""
    try:
        r = requests.get(host + "/api/v3/klines",
                         params=dict(symbol=symbol, interval="1h",
                                     startTime=start_ms, limit=limit),
                         timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
    except Exception:
        return None
    return None


def hourly_from_host(host: str, symbol: str, start_ms: int,
                     end_ms: int) -> dict[int, tuple]:
    """Every hourly kline in ``[start_ms, end_ms)`` from ONE host, keyed by open
    time in ms → ``(open, high, low, close, volume)``.

    One host serves the WHOLE series: paging across hosts on the first error
    splices two venues' volume scales into one column without any trace in the
    data."""
    out: dict[int, tuple] = {}
    cursor = start_ms
    while cursor < end_ms:
        batch = klines_page(host, symbol, cursor)
        if not batch or not isinstance(batch[-1], (list, tuple)):
            break
        for k in batch:
            t = int(k[0])
            if start_ms <= t < end_ms:
                out[t] = tuple(float(k[i]) for i in (1, 2, 3, 4, 5))
        cursor = int(batch[-1][0]) + HOUR_MS
        time.sleep(0.05)
    return out


def fetch_hourly(symbol: str, start_ms: int, end_ms: int,
                 hosts: tuple[str, ...] = BINANCE_HOSTS):
    """``(hours, primary_host)`` — the whole hourly series from the FIRST host
    that answers, so a single venue owns every row.  ``({}, None)`` when no
    host answers."""
    for host in hosts:
        got = hourly_from_host(host, symbol, start_ms, end_ms)
        if got:
            return got, host
    return {}, None


def bar_start_ms(t_ms: int) -> int:
    """The 12:00-UTC bar (its START date, midnight-aligned ms) that an hourly
    open time belongs to.  Bar D spans ``[D 12:00 UTC, D+1 12:00 UTC)``."""
    return ((t_ms - ANCHOR_HOUR_UTC * HOUR_MS) // DAY_MS) * DAY_MS


def bar_hours(bar_ms: int) -> list[int]:
    """The 24 hourly open times a bar is made of."""
    return [bar_ms + (ANCHOR_HOUR_UTC + i) * HOUR_MS for i in range(24)]


def missing_hours(hours: dict[int, tuple], now_ms: int,
                  lookback_days: int = HEAL_LOOKBACK_DAYS) -> list[int]:
    """Open times absent from RECENT bars whose 24-hour window has already
    closed — exactly the hours whose absence drops a completed bar.

    Bars still in progress are skipped (they are *meant* to be incomplete), and
    so are bars older than ``lookback_days``: those rows are pinned history
    under the vintage freeze, so inserting one now would restate a committed
    vintage.  A bar the primary venue barely traded (< ``HEAL_MIN_PRIMARY_HOURS``
    hours) is skipped too — healing patches a hole in a bar the venue has,
    it does not import a whole bar from somewhere else."""
    newest_closed = ((now_ms - (ANCHOR_HOUR_UTC + 24) * HOUR_MS) // DAY_MS) * DAY_MS
    missing: list[int] = []
    for i in range(lookback_days + 1):
        bar = newest_closed - i * DAY_MS
        want = bar_hours(bar)
        have = [t for t in want if t in hours]
        if len(have) == 24 or len(have) < HEAL_MIN_PRIMARY_HOURS:
            continue
        missing += [t for t in want if t not in hours]
    return sorted(missing)


def volume_scale(primary: dict[int, tuple], donor: dict[int, tuple],
                 since_ms: int) -> float | None:
    """The factor that puts a donor's klines on the primary venue's volume
    scale: the median ``primary/donor`` volume ratio over the hours both serve.

    ``None`` — refuse the donor — when the two disagree on PRICE (a different
    symbol or a broken feed, not merely a thinner venue) or when too few hours
    overlap to calibrate."""
    common = [t for t in donor if t in primary and t >= since_ms]
    px = [primary[t][3] / donor[t][3] for t in common if donor[t][3] > 0]
    vol = [primary[t][4] / donor[t][4] for t in common
           if donor[t][4] > 0 and primary[t][4] > 0]
    if len(px) < HEAL_MIN_OVERLAP or len(vol) < HEAL_MIN_OVERLAP:
        return None
    if abs(float(np.median(px)) - 1.0) > HEAL_MAX_PX_DIVERGENCE:
        return None
    k = float(np.median(vol))
    return k if np.isfinite(k) and k > 0 else None


def heal_hourly_gaps(hours: dict[int, tuple], symbol: str, primary_host: str | None,
                     now_ms: int, hosts: tuple[str, ...] = BINANCE_HOSTS,
                     log=print) -> dict[int, tuple]:
    """Fill the hours a completed bar is missing from the next host that has
    them (see the healing rule above).  Returns ``hours``, mutated in place;
    a no-op when nothing is missing or no donor qualifies.

    ``log`` receives one line per healing decision; pass ``None`` to silence it
    (the Streamlit app renders its own warning instead)."""
    say = log if callable(log) else (lambda _msg: None)
    missing = missing_hours(hours, now_ms)
    if not missing:
        return hours
    say(f"  [gap] {symbol}: {len(missing)} hour(s) missing from completed "
        f"12:00-UTC bar(s) on {primary_host} — trying the other hosts")
    since = min(missing) - 30 * DAY_MS
    for host in hosts:
        if host == primary_host:
            continue
        donor = hourly_from_host(host, symbol, since, now_ms)
        if not donor:
            continue
        k = volume_scale(hours, donor, since)
        if k is None:
            say(f"  [gap] {symbol}: {host} REFUSED — prices disagree or the "
                f"volume scale could not be calibrated")
            continue
        filled = 0
        for t in missing:
            if t in donor and t not in hours:
                o, h, l_, c, v = donor[t]
                hours[t] = (o, h, l_, c, v * k)
                filled += 1
        say(f"  [gap] {symbol}: {filled}/{len(missing)} hour(s) filled from "
            f"{host} (volume rescaled x{k:.6g} onto {primary_host}'s scale)")
        missing = missing_hours(hours, now_ms)
        if not missing:
            return hours
    say(f"  [gap] {symbol}: {len(missing)} hour(s) still missing — the "
        f"affected completed bar(s) stay dropped (the freshness audit will "
        f"flag the sleeve stale rather than publish a spliced bar)")
    return hours


def heal_hourly_frame(h: pd.DataFrame, symbol: str, primary_host: str | None = None,
                      now_ms: int | None = None, hosts: tuple[str, ...] = BINANCE_HOSTS,
                      log=print) -> pd.DataFrame:
    """``heal_hourly_gaps`` for callers holding an hourly OHLCV DataFrame
    (UTC-naive index).  Only the healing window is considered, and only rows
    the frame does not already have are added — an existing row always wins, so
    the primary venue's own klines are never overwritten by a donor's."""
    if h is None or h.empty:
        return h
    if now_ms is None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    # Enough history for `volume_scale` to calibrate on 30 days of overlap.
    window_start = bar_start_ms(now_ms) - (HEAL_LOOKBACK_DAYS + 31) * DAY_MS
    recent = h.loc[h.index >= pd.Timestamp(window_start, unit="ms")]
    if recent.empty:
        return h
    hours = {int(ts.timestamp() * 1000): tuple(float(v) for v in row)
             for ts, row in zip(recent.index, recent[_OHLCV].to_numpy())}
    before = set(hours)
    hours = heal_hourly_gaps(hours, symbol, primary_host, now_ms,
                             hosts=hosts, log=log)
    added = {t: v for t, v in hours.items() if t not in before}
    if not added:
        return h
    add = pd.DataFrame.from_dict(added, orient="index", columns=_OHLCV)
    add.index = pd.to_datetime(pd.Index(add.index), unit="ms")
    out = pd.concat([h, add.reindex(columns=h.columns)])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = h.index.name
    return out


def rebucket_12utc(hourly: pd.DataFrame) -> pd.DataFrame:
    """Group hourly OHLCV into 24h bars starting at ``ANCHOR_HOUR_UTC``,
    indexed by bar-START date.  Bars with anything other than 24 hours are
    DROPPED — run ``heal_hourly_frame`` first so a host's hourly gap does not
    cost an otherwise-complete completed bar."""
    if hourly is None or hourly.empty:
        g = pd.DataFrame(columns=_OHLCV)
        g.index = pd.DatetimeIndex([], name="bar_start")
        return g
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), n_hours=("close", "size"),
    )
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "bar_start"
    return g
