"""Interior-gap repair for the live 12:00-UTC bar builder.

``_rebucket_12utc`` (app/btc_hourly_app.py) keeps a daily bar only when all 24
of its hourly klines are present, so a single missing hour DROPS an otherwise
complete completed bar and every DAILY view — H/L forecast, CT signal, position
— silently falls back to the previous bar.  ``api.binance.us`` (the first host
tried) periodically serves NO kline for a run of hours, which is exactly how the
page came to report "Daily signal bar is 2 bar(s) behind".

The offline puller already heals this (``scripts/pull_backtest_data.py``), but
the live app rebuilds its own daily bars from its own hourly pull and had no
equivalent, and its only fallback trigger was a STALE TAIL — ``max(index)`` more
than three hours old.  An interior hole leaves the tail perfectly current, so
that guard never fired for the one failure mode that drops a bar.

Healing rule (mirrors the puller, on the app's DataFrame representation):

* only hours of bars whose 24-hour window has already CLOSED are missing —
  a bar still in progress is *meant* to be short;
* only the recent window (``HEAL_LOOKBACK_DAYS``), which is all the daily views
  read;
* only a bar the primary venue actually traded (>= ``HEAL_MIN_PRIMARY_HOURS``
  of its own hours) — this patches a hole, it never imports a whole bar from
  somewhere else;
* a donor must AGREE ON PRICE with the primary (median |close ratio - 1| within
  ``HEAL_MAX_PX_DIVERGENCE``) over at least ``HEAL_MIN_OVERLAP`` shared hours,
  and its volume is rescaled by the median volume ratio onto the primary
  venue's scale.  Venues agree on price to ~0.01% but their raw volumes differ
  by a couple of orders of magnitude, and the model's volume features
  (``vol_chg_1``, ``vol_z_20``, ``vol_ma_ratio``) would read an unscaled venue
  switch as a volume shock lasting the whole 20-bar window.

A donor that cannot be calibrated is REFUSED: the bar is then dropped exactly as
before and the page's freshness banner correctly reports the sleeve behind,
rather than the daily views running on a spliced bar.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

ANCHOR_HOUR_UTC        = 12     # bar D spans [D 12:00 UTC, D+1 12:00 UTC)
HEAL_LOOKBACK_DAYS     = 5      # the recent window every daily view reads
HEAL_MIN_PRIMARY_HOURS = 12     # patch a hole in a bar the venue traded
HEAL_MIN_OVERLAP       = 48     # hours both feeds must share to calibrate volume
HEAL_MAX_PX_DIVERGENCE = 0.01   # 1% median |close ratio - 1| tolerated

OHLCV = ["open", "high", "low", "close", "volume"]


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def bar_start(ts) -> pd.Timestamp:
    """The 12:00-UTC bar (its START date) an hourly open time belongs to."""
    return (pd.Timestamp(ts) - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()


def bar_hours(bar: pd.Timestamp) -> pd.DatetimeIndex:
    """The 24 hourly open times a bar is made of."""
    first = pd.Timestamp(bar) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
    return pd.date_range(first, periods=24, freq="h")


def newest_closed_bar(now=None) -> pd.Timestamp:
    """Start date of the newest bar whose 24-hour window has already closed."""
    now = _utcnow() if now is None else pd.Timestamp(now)
    return bar_start(now - pd.Timedelta(hours=24))


def _present(hourly: pd.DataFrame) -> pd.DatetimeIndex:
    """Hours the frame really carries — an all-NaN row is a hole, not a bar."""
    if hourly.empty:
        return pd.DatetimeIndex([])
    col = "close" if "close" in hourly.columns else hourly.columns[0]
    return hourly.index[hourly[col].notna()]


def missing_hours(hourly: pd.DataFrame, now=None,
                  lookback_days: int = HEAL_LOOKBACK_DAYS) -> pd.DatetimeIndex:
    """Open times absent from recent ALREADY-CLOSED bars — exactly the hours
    whose absence drops a completed bar from ``_rebucket_12utc``."""
    have = set(_present(hourly))
    newest = newest_closed_bar(now)
    missing: list[pd.Timestamp] = []
    for i in range(lookback_days + 1):
        want = bar_hours(newest - pd.Timedelta(days=i))
        here = [t for t in want if t in have]
        if len(here) == 24 or len(here) < HEAL_MIN_PRIMARY_HOURS:
            continue
        missing += [t for t in want if t not in have]
    return pd.DatetimeIndex(sorted(missing))


def volume_scale(primary: pd.DataFrame, donor: pd.DataFrame,
                 since: pd.Timestamp) -> float | None:
    """Factor putting a donor's klines on the primary feed's volume scale: the
    median ``primary/donor`` volume ratio over the hours both serve.

    ``None`` — refuse the donor — when the two disagree on PRICE (a different
    feed, not merely a thinner venue) or when too few hours overlap."""
    common = primary.index.intersection(donor.index)
    common = common[common >= pd.Timestamp(since)]
    if len(common) < HEAL_MIN_OVERLAP:
        return None
    p, d = primary.loc[common], donor.loc[common]
    px = (p["close"] / d["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    ok = (p["volume"] > 0) & (d["volume"] > 0)
    vol = (p["volume"][ok] / d["volume"][ok]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < HEAL_MIN_OVERLAP or len(vol) < HEAL_MIN_OVERLAP:
        return None
    if abs(float(px.median()) - 1.0) > HEAL_MAX_PX_DIVERGENCE:
        return None
    k = float(vol.median())
    return k if np.isfinite(k) and k > 0 else None


def heal_hourly_gaps(primary: pd.DataFrame, donors, now=None) -> pd.DataFrame:
    """Fill the hours a recent COMPLETED bar is missing from the first donor
    that qualifies, and return the repaired frame.

    ``donors`` is an iterable of ``(label, source)`` where ``source`` is either
    a DataFrame or a zero-argument callable returning one — callables are only
    invoked when there is actually a gap to fill, so the common (healthy) case
    costs no extra network round-trip.  A no-op when nothing is missing or no
    donor qualifies; never overwrites an hour the primary already has."""
    missing = missing_hours(primary, now)
    if not len(missing):
        return primary

    healed = primary.copy()
    since = missing.min() - pd.Timedelta(days=30)
    for _label, source in donors:
        try:
            donor = source() if callable(source) else source
        except Exception:
            continue
        if donor is None or donor.empty:
            continue
        donor = donor[~donor.index.duplicated(keep="last")].sort_index()
        k = volume_scale(healed, donor, since)
        if k is None:
            continue                       # prices disagree / not calibratable
        take = donor.index.intersection(missing).difference(_present(healed))
        if not len(take):
            continue
        patch = donor.loc[take, OHLCV].copy()
        patch["volume"] = patch["volume"] * k
        # Keep the primary's exact schema: a donor carries only OHLCV, so
        # concat would otherwise reorder (or union) the columns of a frame the
        # bar builder aggregates by name.
        patch = patch.reindex(columns=healed.columns)
        healed = pd.concat([healed.drop(index=take, errors="ignore"), patch])
        healed = healed[~healed.index.duplicated(keep="last")].sort_index()
        missing = missing_hours(healed, now)
        if not len(missing):
            break
    return healed
