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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


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
