"""Unit tests for the split-safety guards added after the 2026-08 MSTU splice.

A reverse split restates every historical price at once.  The vintage freeze
pinned the OLD scale while newly pulled rows arrived on the NEW one, splicing
two scales into one column and fabricating a +917.81% single-day gain in
data/backtest/mstu_synthetic_daily.csv (MSTU really moved +5.71%).  Nothing in
the pull or the validator stopped it.

These tests cover both halves of the fix:
  * pull_backtest_data._split_factor / _freeze_history — rescale the pinned
    vintage onto a split, but never onto drifting or corrupted data;
  * validate_refreshed_data._global_scale / check_frozen_history /
    check_no_new_jumps — accept a whole-history rescale, reject anything that
    changes what happened on a day, and block a fabricated jump outright.

The final test replays the real failure end to end and asserts it is caught.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pull = _load("pull_backtest_data")
val = _load("validate_refreshed_data")

IDX = pd.bdate_range("2024-01-01", periods=300)


def _series(seed=3, n=300):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, n))
    return pd.Series(px, index=IDX[:n], name="close")


# ── pull-side: _split_factor ────────────────────────────────────────────────

def test_split_factor_finds_a_clean_whole_history_rescale():
    pinned = _series()
    fresh = pinned * 9.9002            # the provider restates for a split
    assert pull._split_factor(pinned, fresh) == pytest.approx(9.9002, rel=1e-9)


def test_split_factor_ignores_an_unchanged_series():
    s = _series()
    assert pull._split_factor(s, s) is None


def test_split_factor_refuses_drifting_data():
    # a rescale that only holds for part of the history is corruption, not a
    # corporate action — waving it through is precisely the MSTU bug
    pinned = _series()
    fresh = pinned * 9.9002
    fresh.iloc[-20:] = fresh.iloc[-20:] * 1.4
    assert pull._split_factor(pinned, fresh) is None


def test_split_factor_refuses_too_little_overlap():
    pinned = _series(n=40)
    assert pull._split_factor(pinned, pinned * 9.9) is None


# ── pull-side: the freeze actually applies it ───────────────────────────────

def test_freeze_rescales_the_pinned_vintage_on_a_split(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    pinned = _series()
    csv = tmp_path / "pinned.csv"
    pinned.to_frame().to_csv(csv)
    # fresh pull: whole history on the new scale, plus one appended day
    fresh = (pinned * 9.9002).to_frame()
    nxt = IDX[len(pinned)] if len(IDX) > len(pinned) else pinned.index[-1] + pd.Timedelta(days=3)
    fresh.loc[nxt, "close"] = float(fresh["close"].iloc[-1]) * 1.01

    out = pull._freeze_history("t", fresh, csv, tail=0, split_safe=True)
    # the pinned rows survive on the NEW scale — returns intact, no splice
    assert np.allclose(out["close"].reindex(pinned.index),
                       pinned * 9.9002, rtol=1e-9)
    assert float(np.log(out["close"] / out["close"].shift(1)).abs().max()) < np.log(1.8)
    assert "rescaled" in capsys.readouterr().out


def test_freeze_without_split_safe_reproduces_the_bug(tmp_path, monkeypatch):
    # the pre-fix behaviour, kept as a regression witness
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    pinned = _series()
    csv = tmp_path / "pinned.csv"
    pinned.to_frame().to_csv(csv)
    fresh = (pinned * 9.9002).to_frame()
    nxt = pinned.index[-1] + pd.Timedelta(days=3)
    fresh.loc[nxt, "close"] = float(fresh["close"].iloc[-1]) * 1.01
    out = pull._freeze_history("t", fresh, csv, tail=0, split_safe=False)
    jump = float(np.log(out["close"] / out["close"].shift(1)).max())
    assert jump > np.log(1.8)          # old scale spliced to new → fake gain


# ── validator-side ─────────────────────────────────────────────────────────

def test_frozen_check_accepts_a_split_only_when_scale_invariant():
    old = _series().to_frame()
    new = (old * 9.9002)
    assert val.check_frozen_history(new, old, tail=0) != []
    assert val.check_frozen_history(new, old, tail=0, scale_invariant=True) == []


def test_frozen_check_still_rejects_a_single_restated_bar():
    old = _series().to_frame()
    new = (old * 9.9002)
    new.iloc[100, 0] *= 1.05           # one bar moved relative to the rest
    assert val.check_frozen_history(new, old, tail=0, scale_invariant=True) != []


def test_jump_check_blocks_a_fabricated_splice():
    px = _series()
    spliced = px.copy()
    spliced.iloc[-5:] = spliced.iloc[-5:] * 9.9002      # scale break at the tail
    errs = val.check_no_new_jumps(spliced.to_frame(), px.to_frame(), "close")
    assert errs and "corrupted splice" in errs[0]


def test_jump_check_ignores_a_move_already_committed():
    px = _series()
    px.iloc[150:] = px.iloc[150:] * 3.0                 # pre-existing anomaly
    assert val.check_no_new_jumps(px.to_frame(), px.to_frame(), "close") == []


def test_jump_check_passes_ordinary_history():
    px = _series()
    assert val.check_no_new_jumps(px.to_frame(), px.to_frame(), "close") == []


# ── the real failure, replayed ─────────────────────────────────────────────

def test_the_real_mstu_splice_would_now_be_caught():
    syn = pd.read_csv(ROOT / "data/backtest/mstu_synthetic_daily.csv",
                      index_col=0, parse_dates=True,
                      float_precision="round_trip")
    # the shipped file is clean
    assert val.check_no_new_jumps(syn, None, "close") == []
    # re-create the splice: put the pre-split scale back on everything before
    # the corporate action and leave the tail on the new one
    broken = syn.copy()
    cut = broken.index < pd.Timestamp("2026-08-24")
    broken.loc[cut, "close"] = broken.loc[cut, "close"] / 9.900208076
    errs = val.check_no_new_jumps(broken, syn, "close")
    assert errs, "the splice that shipped must not pass the guard"
    assert "+917" in errs[0] or "corrupted splice" in errs[0]
