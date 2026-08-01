"""Unit tests for overall_core.live_entry_keys — the green mirror of
live_exit_keys used by the Today's-action-plan table.

A flat trend instrument with no committed buy signal whose LIVE price
satisfies the mode's real long condition would signal at today's close and
open next bar — the table flags it green ("likely enters next bar"), exactly
like a broken-trend hold is flagged red ("exits next bar").
"""
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402
import ticker_config as tcfg  # noqa: E402


def _ma_cfg(win=3):
    return SimpleNamespace(strategy_mode="ma", ma_window=win)


def _dual_cfg(fast=2, slow=4):
    return SimpleNamespace(strategy_mode="dual_ma", ma_fast=fast, ma_slow=slow)


def _res(key="XLE", mode="ma", in_pos=False, tone="flat", cfg=None,
         close_hist=None, ma_val=None, parent=None):
    return dict(key=key, mode=mode, parent=parent or key,
                pos=dict(in_pos=in_pos), decision=dict(tone=tone),
                cfg=cfg, close_hist=close_hist, ma_val=ma_val)


def test_flat_name_flips_long_on_live_price():
    # last closes are below the SMA (flat), but a strong live print takes the
    # provisional newest close above it → likely enters next bar.
    r = _res(cfg=_ma_cfg(3), close_hist=[100.0, 90.0, 80.0])
    assert oc.live_entry_keys([r], {"XLE": {"price": 200.0}}) == {"XLE"}


def test_flat_name_still_below_trend_not_flagged():
    r = _res(cfg=_ma_cfg(3), close_hist=[100.0, 90.0, 80.0])
    assert oc.live_entry_keys([r], {"XLE": {"price": 10.0}}) == set()


def test_committed_buy_and_open_positions_are_excluded():
    # tone "buy" = the last close already fired the entry (action OPEN) — a
    # committed entry, not a live "likely"; in_pos = nothing to enter.
    buy = _res(key="A", tone="buy", cfg=_ma_cfg(3), close_hist=[1.0, 1.0, 1.0])
    held = _res(key="B", in_pos=True, tone="hold", cfg=_ma_cfg(3),
                close_hist=[1.0, 1.0, 1.0])
    spot = {"A": {"price": 100.0}, "B": {"price": 100.0}}
    assert oc.live_entry_keys([buy, held], spot) == set()


def test_dual_ma_needs_a_real_golden_cross_not_price_above_slow_sma():
    # price pops above the slow SMA but the fast SMA is still below it — the
    # real dual_ma long condition (fast > slow) has NOT flipped, so a naive
    # price-vs-line read would over-flag here.
    r = _res(mode="dual_ma", cfg=_dual_cfg(2, 4),
             close_hist=[100.0, 90.0, 80.0, 70.0])
    assert oc.live_entry_keys([r], {"XLE": {"price": 75.0}}) == set()
    # a print strong enough to golden-cross the SMAs does flag it.
    assert oc.live_entry_keys([r], {"XLE": {"price": 300.0}}) == {"XLE"}


def test_plain_ma_fallback_without_cfg_uses_price_vs_sma():
    # stale-cache fallback: no cfg/close_hist, plain `ma` only — the naive
    # price-vs-SMA read IS the real entry rule there.
    r = _res(cfg=None, close_hist=None, ma_val=50.0)
    assert oc.live_entry_keys([r], {"XLE": {"price": 60.0}}) == {"XLE"}
    assert oc.live_entry_keys([r], {"XLE": {"price": 40.0}}) == set()


def test_missing_live_quote_and_non_trend_modes_skip():
    no_quote = _res(key="A", cfg=_ma_cfg(3), close_hist=[1.0, 1.0, 1.0])
    macd = _res(key="B", mode="macd", cfg=None, close_hist=None)
    div = _res(key="C", mode="divergence", cfg=None, close_hist=None)
    spot = {"B": {"price": 1.0}, "C": {"price": 1.0}}
    assert oc.live_entry_keys([no_quote, macd, div], spot) == set()


# ── divergence apps (e.g. ARTY) — provisional live-bar entry gate ──────────
def _div_fixture(n=120):
    """Synthetic completed-signature frame: quiet errors throughout, then the
    last TWO completed bars post high-breaks (err_hi ≈ +0.8 vs a ~0 median) —
    one more high-break at today's provisional bar fires ARTY's U1 (err_hi
    3d-avg > 0.32 with ≥2 high-breaks) inside a bull regime (rising closes)."""
    rows = []
    for i in range(n):
        c = 100.0 + 0.1 * i
        ph, pl = c * 1.01, c * 0.99
        ah = ph + (0.008 * c if i >= n - 2 else 0.0)
        rows.append(dict(close_asof=c, pred_high=ph, pred_low=pl,
                         actual_high=ah, actual_low=pl,
                         target_date=pd.Timestamp("2026-01-01")
                         + pd.Timedelta(days=i)))
    completed = pd.DataFrame(rows)
    c0 = 100.0 + 0.1 * n
    pending = dict(close_asof=c0, pred_high=c0 * 1.01, pred_low=c0 * 0.99)
    return completed, pending, c0


def _div_res(completed, pending, tone="watch"):
    return dict(key="ARTY", mode="divergence", parent="ARTY",
                pos=dict(in_pos=False), decision=dict(tone=tone),
                cfg=tcfg.CONFIGS["ARTY"], close_hist=None, ma_val=None,
                sig_completed=completed, pending_pred=pending)


def test_divergence_strong_live_rally_fires_the_entry_gate():
    completed, pending, c0 = _div_fixture()
    r = _div_res(completed, pending)
    live = pending["pred_high"] * 1.01          # live px breaks the model band
    assert oc.live_entry_keys([r], {"ARTY": {"price": live}}) == {"ARTY"}


def test_divergence_weak_live_price_not_flagged():
    completed, pending, c0 = _div_fixture()
    r = _div_res(completed, pending)
    live = pending["pred_low"] * 0.99           # live px sinks below the band
    assert oc.live_entry_keys([r], {"ARTY": {"price": live}}) == set()


def test_divergence_without_pending_bands_or_history_skips():
    completed, pending, c0 = _div_fixture()
    no_bands = _div_res(completed, None)
    no_hist = _div_res(None, pending)
    spot = {"ARTY": {"price": pending["pred_high"] * 1.01}}
    assert oc.live_entry_keys([no_bands, no_hist], spot) == set()


# ── stale re-served quotes (weekends / holidays / after-hours) ──────────────
# Yahoo's regularMarketPrice repeats the last session close when the market is
# shut. Appending that price as a provisional NEW bar duplicates the last
# close and shifts every rolling window by one — flipping marginal dual-MA
# crosses and manufacturing phantom flags that contradict the committed
# decision (and the ticker app). Observed: REMX "exits next bar" on a
# Saturday while its app showed LONG — HOLDING. The live overlay must stay
# silent until a genuinely new price exists.
def test_weekend_requote_never_phantom_flags_an_exit():
    # committed dual-MA state is LONG (fast 27.5 > slow 21.67), but appending
    # the last close AGAIN flips the shifted SMAs (25 < 26.67) — the exact
    # phantom-exit mechanism. Live == last close → no new info → not flagged.
    r = _res(key="REMX", mode="dual_ma", in_pos=True, tone="hold",
             cfg=_dual_cfg(2, 3), close_hist=[10.0, 30.0, 25.0])
    assert oc.live_exit_keys([r], {"REMX": {"price": 25.0}}) == set()
    # …a quote within 1 bp of the close is the same non-information
    assert oc.live_exit_keys([r], {"REMX": {"price": 25.001}}) == set()
    # …but a genuinely lower live print still flags the real cross-break
    assert oc.live_exit_keys([r], {"REMX": {"price": 20.0}}) == {"REMX"}


def test_weekend_requote_never_phantom_flags_an_entry():
    # committed dual-MA state is FLAT (fast 12.5 < slow 18.33); duplicating
    # the last close would golden-cross the shifted SMAs (15 > 13.33) — the
    # phantom-entry mirror. Live == last close → not flagged.
    r = _res(key="GRID", mode="dual_ma", cfg=_dual_cfg(2, 3),
             close_hist=[30.0, 10.0, 15.0])
    assert oc.live_entry_keys([r], {"GRID": {"price": 15.0}}) == set()
    # …a genuinely higher live print still flags the real golden cross
    assert oc.live_entry_keys([r], {"GRID": {"price": 100.0}}) == {"GRID"}


def test_weekend_requote_skips_the_divergence_entry_too():
    # same rule for divergence apps (e.g. ARTY): a re-served last close is
    # not a live rally — no provisional-bar entry check until a new price
    completed, pending, c0 = _div_fixture()
    strong = pending["pred_high"] * 1.01
    r = _div_res(completed, pending)
    r["close_hist"] = [c0 * 0.99, strong]       # last committed close
    assert oc.live_entry_keys([r], {"ARTY": {"price": strong}}) == set()
    # …a genuinely new price above the band still fires the gate
    r2 = _div_res(completed, pending)
    r2["close_hist"] = [c0 * 0.99, c0]
    assert oc.live_entry_keys([r2], {"ARTY": {"price": strong}}) == {"ARTY"}
