"""Unit tests for the shared 12:00-UTC bar builder (``binance_bars``).

The BTC/MSTR/MSTU/ETH sleeve reads a committed daily CSV, and a daily bar is
only written when all 24 of its hourly klines are present.  `api.binance.us`
(the first host tried) periodically serves NO kline for a run of hours, which
silently dropped an otherwise-complete bar — the sleeve then froze a bar behind
and the publisher's freshness audit withheld the day's target book
(observed 2026-08-31: the 08-30 bar was missing 8 of its 24 hours).

The healing first landed in the puller alone, which left the LIVE app dropping
exactly the bars the dataset had learned to recover: on 2026-08-31 the page
warned "daily signal bar is 1 bar behind" all afternoon while
raw_features_daily.csv already carried the 08-30 bar.  Both now build their
bars from ``binance_bars``, and these tests pin all three parts of the defect:

* the healing contract — gaps in an ALREADY-CLOSED recent bar are filled from
  another host with the donor's volume rescaled onto the primary venue's scale,
  while bars in progress, pinned history and donors that disagree are left
  alone;
* the app building its hourly frame through that same healing, on the same host
  list, before anything rebuckets it;
* the BTC app's freshness caption — it must report the bar it HOLDS, not the
  newest hourly timestamp, or the page (and the Daily Audit record it writes)
  claims a close it has no bar for.

Fully offline — every host fetch is monkeypatched.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import binance_bars as B  # noqa: E402
import pull_backtest_data as P  # noqa: E402

HOUR = B.HOUR_MS
DAY = B.DAY_MS
VENUE_RATIO = 300.0          # donor (deep venue) volume ÷ primary (thin venue)


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def _close(t: int) -> float:
    """Deterministic price walk — identical on both venues (they agree to ~0.01%)."""
    return 80_000.0 + (t // HOUR) % 97


def _series(now_ms: int, days: int = 40, scale: float = 1.0,
            px_factor: float = 1.0, drop: tuple = ()) -> dict:
    """Synthetic hourly klines for the `days` before `now_ms`, minus `drop`."""
    first = B.bar_start_ms(now_ms) - days * DAY + B.ANCHOR_HOUR_UTC * HOUR
    out = {}
    t = first
    while t < now_ms:
        if t not in drop:
            c = _close(t) * px_factor
            out[t] = (c, c * 1.001, c * 0.999, c, (1.0 + (t // HOUR) % 5) * scale)
        t += HOUR
    return out


def _newest_closed_bar(now_ms: int) -> int:
    return ((now_ms - (B.ANCHOR_HOUR_UTC + 24) * HOUR) // DAY) * DAY


# ── which hours count as "missing" ───────────────────────────────────────────
def test_no_gaps_means_nothing_to_heal():
    now = _now_ms()
    assert B.missing_hours(_series(now), now) == []


def test_gap_in_a_closed_bar_is_reported():
    now = _now_ms()
    bar = _newest_closed_bar(now)
    holes = tuple(B.bar_hours(bar)[4:12])          # 8 hours, as on 2026-08-31
    assert B.missing_hours(_series(now, drop=holes), now) == sorted(holes)


def test_bar_still_in_progress_is_not_healed():
    now = _now_ms()
    # the bar AFTER the newest closed one is open by construction; its hours
    # after `now` simply do not exist yet and must not be treated as a gap
    open_bar = _newest_closed_bar(now) + DAY
    hours = _series(now)
    assert not [t for t in B.missing_hours(hours, now) if B.bar_start_ms(t) == open_bar]


def test_pinned_history_beyond_the_lookback_is_left_alone():
    now = _now_ms()
    old = _newest_closed_bar(now) - (B.HEAL_LOOKBACK_DAYS + 2) * DAY
    holes = tuple(B.bar_hours(old)[:3])
    assert B.missing_hours(_series(now, days=40, drop=holes), now) == []


def test_a_bar_the_venue_barely_traded_is_not_imported_wholesale():
    now = _now_ms()
    bar = _newest_closed_bar(now)
    holes = tuple(B.bar_hours(bar)[:20])           # only 4 primary hours left
    assert B.missing_hours(_series(now, drop=holes), now) == []


# ── healing itself ──────────────────────────────────────────────────────────
def _patch_donor(monkeypatch, donor: dict):
    def fake(host, symbol, start_ms, end_ms):
        return {t: v for t, v in donor.items() if start_ms <= t < end_ms}
    monkeypatch.setattr(B, "hourly_from_host", fake)


def test_gap_is_filled_and_volume_rescaled_onto_the_primary_scale(monkeypatch):
    now = _now_ms()
    bar = _newest_closed_bar(now)
    holes = tuple(B.bar_hours(bar)[4:12])
    primary = _series(now, drop=holes)
    _patch_donor(monkeypatch, _series(now, scale=VENUE_RATIO))
    healed = B.heal_hourly_gaps(dict(primary), "BTCUSDT", B.BINANCE_HOSTS[0], now)

    assert B.missing_hours(healed, now) == []
    for t in holes:
        assert healed[t][3] == pytest.approx(_close(t))                  # donor's price
        assert healed[t][4] == pytest.approx((1.0 + (t // HOUR) % 5), rel=1e-6)  # primary's scale
    # the healed hours must not tower over the venue's own volumes
    own = [v[4] for t, v in primary.items() if B.bar_start_ms(t) == bar]
    assert max(healed[t][4] for t in holes) <= max(own) * 1.5


def test_donor_whose_prices_disagree_is_refused(monkeypatch):
    now = _now_ms()
    holes = tuple(B.bar_hours(_newest_closed_bar(now))[4:12])
    primary = _series(now, drop=holes)
    _patch_donor(monkeypatch, _series(now, scale=VENUE_RATIO, px_factor=1.2))
    healed = B.heal_hourly_gaps(dict(primary), "BTCUSDT", B.BINANCE_HOSTS[0], now)
    assert B.missing_hours(healed, now) == sorted(holes)   # bar stays dropped


def test_donor_with_too_little_overlap_is_refused(monkeypatch):
    now = _now_ms()
    holes = tuple(B.bar_hours(_newest_closed_bar(now))[4:12])
    primary = _series(now, drop=holes)
    thin = _series(now, scale=VENUE_RATIO)
    keep = set(holes) | set(sorted(thin)[-B.HEAL_MIN_OVERLAP + 1:])
    _patch_donor(monkeypatch, {t: v for t, v in thin.items() if t in keep})
    healed = B.heal_hourly_gaps(dict(primary), "BTCUSDT", B.BINANCE_HOSTS[0], now)
    assert B.missing_hours(healed, now) == sorted(holes)


# ── end to end: the completed bar reaches the daily frame ───────────────────
def test_fetch_12utc_recovers_the_bar_a_host_gap_would_have_dropped(monkeypatch):
    now = _now_ms()
    bar = _newest_closed_bar(now)
    holes = tuple(B.bar_hours(bar)[4:12])
    primary = _series(now, drop=holes)
    donor = _series(now, scale=VENUE_RATIO)

    def fake(host, symbol, start_ms, end_ms):
        src = primary if host == B.BINANCE_HOSTS[0] else donor
        return {t: v for t, v in src.items() if start_ms <= t < end_ms}
    monkeypatch.setattr(B, "hourly_from_host", fake)

    start = pd.Timestamp(now - 12 * DAY, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    g = P.fetch_12utc(start)
    bar_date = pd.Timestamp(bar, unit="ms")
    assert bar_date in g.index                                  # the healed bar is there
    assert g.loc[bar_date, "close"] == pytest.approx(_close(B.bar_hours(bar)[-1]))
    assert g.index.max() == bar_date                            # no in-progress bar leaks in


# ── the DataFrame door the live app comes in through ────────────────────────
def _frame(hours: dict) -> pd.DataFrame:
    h = pd.DataFrame.from_dict(hours, orient="index",
                               columns=["open", "high", "low", "close", "volume"])
    h.index = pd.to_datetime(pd.Index(h.index), unit="ms")
    h.index.name = "ts"
    return h.sort_index()


def test_heal_hourly_frame_recovers_the_daily_bar_the_app_would_have_dropped(monkeypatch):
    """The live path in one line: an hourly frame with a hole in a CLOSED bar
    loses that bar to `rebucket_12utc` (that IS the "1 bar behind" warning), and
    healing the frame first brings it back on the primary venue's volume scale."""
    now = _now_ms()
    bar = _newest_closed_bar(now)
    holes = tuple(B.bar_hours(bar)[4:12])           # 8 hours, as on 2026-08-31
    primary = _frame(_series(now, drop=holes))
    bar_date = pd.Timestamp(bar, unit="ms")

    assert bar_date not in B.rebucket_12utc(primary).index      # the defect

    _patch_donor(monkeypatch, _series(now, scale=VENUE_RATIO))
    healed = B.heal_hourly_frame(primary, "BTCUSDT", B.BINANCE_HOSTS[0],
                                 now_ms=now, log=None)
    bars = B.rebucket_12utc(healed)
    assert bar_date in bars.index                               # the fix
    assert bars.loc[bar_date, "close"] == pytest.approx(_close(B.bar_hours(bar)[-1]))
    # summed on the primary venue's scale — not ~300x it
    own = B.rebucket_12utc(_frame(_series(now)))
    assert bars.loc[bar_date, "volume"] == pytest.approx(own.loc[bar_date, "volume"],
                                                         rel=1e-6)


def test_heal_hourly_frame_never_overwrites_the_primary_venue_s_own_hours(monkeypatch):
    now = _now_ms()
    holes = tuple(B.bar_hours(_newest_closed_bar(now))[4:12])
    primary = _frame(_series(now, drop=holes))
    _patch_donor(monkeypatch, _series(now, scale=VENUE_RATIO))
    healed = B.heal_hourly_frame(primary, "BTCUSDT", B.BINANCE_HOSTS[0],
                                 now_ms=now, log=None)
    kept = healed.loc[primary.index]
    pd.testing.assert_frame_equal(kept, primary)


def test_heal_hourly_frame_is_a_noop_when_no_donor_qualifies(monkeypatch):
    now = _now_ms()
    holes = tuple(B.bar_hours(_newest_closed_bar(now))[4:12])
    primary = _frame(_series(now, drop=holes))
    _patch_donor(monkeypatch, _series(now, scale=VENUE_RATIO, px_factor=1.2))
    healed = B.heal_hourly_frame(primary, "BTCUSDT", B.BINANCE_HOSTS[0],
                                 now_ms=now, log=None)
    pd.testing.assert_frame_equal(healed, primary)


# ════════════════════════════════════════════════════════════════════════════
# The BTC app must build its bars through the SAME healed builder
# ════════════════════════════════════════════════════════════════════════════
# Streamlit runs at import, so these read the app's source.  They pin the split
# that caused the second half of the 2026-08-31 incident: the puller healed its
# hourly gaps and the app did not, so the committed dataset was fresh while the
# page stayed a bar behind.
_APP_SRC = (Path(__file__).resolve().parent.parent
            / "app" / "btc_hourly_app.py").read_text()


def test_app_uses_the_shared_host_list():
    assert "BINANCE_API_HOSTS = _bb.BINANCE_HOSTS" in _APP_SRC
    # the donor the healing borrows from must be reachable from the app at all
    assert "https://data-api.binance.vision" in B.BINANCE_HOSTS


def test_app_heals_hourly_gaps_before_rebucketing():
    assert "_bb.heal_hourly_frame(df, \"BTCUSDT\"" in _APP_SRC
    heal_at = _APP_SRC.index("_bb.heal_hourly_frame(df")
    yahoo_at = _APP_SRC.index("yf_df = _fetch_yfinance_hourly_fallback()", heal_at - 4000)
    # calibrating the donor's volume scale against Yahoo rows would be meaningless
    assert heal_at < yahoo_at


def test_app_rebuckets_with_the_shared_builder():
    assert "return _bb.rebucket_12utc(hourly)" in _APP_SRC


def test_app_pins_one_host_for_the_whole_hourly_series():
    assert '_bb.fetch_hourly("BTCUSDT"' in _APP_SRC
    assert '_bb.fetch_hourly("ETHUSDT"' in _APP_SRC


# ════════════════════════════════════════════════════════════════════════════
# The BTC app's freshness caption must report the bar it HOLDS
# ════════════════════════════════════════════════════════════════════════════
# app/btc_hourly_app.py runs Streamlit at import, so this reads the source of
# the caption block instead — enough to pin the contract that broke: as_of came
# from `latest_t_global` (the newest HOURLY candle), so with the daily bar
# dropped for missing hours the page advertised a close it had no bar for, and
# the record it writes outranks every other source in
# freshness.freshest_signal_record — hiding the staleness in the Daily Audit
# tab too, while the Overall app flagged BTC STALE off the same data.
_CAPTION_BLOCK = (
    Path(__file__).resolve().parent.parent / "app" / "btc_hourly_app.py"
).read_text().split("# ── signal-freshness caption")[1].split("\n\n\n")[0]


def test_caption_asof_comes_from_the_daily_bars_not_the_hourly_clock():
    assert "_sig_asof = pd.Timestamp(_fetch_daily_raw().index.max())" in _CAPTION_BLOCK
    asof_lines = [l for l in _CAPTION_BLOCK.splitlines()
                  if "_sig_asof =" in l and not l.strip().startswith("#")]
    assert asof_lines and not any("latest_t_global" in l for l in asof_lines)


def test_caption_warns_and_records_when_the_daily_bar_is_behind():
    assert "_sig_expected = _fr.expected_crypto_asof()" in _CAPTION_BLOCK
    assert "st.warning(" in _CAPTION_BLOCK
    assert "as_of=str(_sig_asof.date())" in _CAPTION_BLOCK
