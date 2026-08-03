"""Unit tests for the BTC-vintage history freeze:

* ``pull_backtest_data._freeze_history`` — committed rows stay pinned; only
  the tail correction window and genuinely new rows take fresh values.
* ``validate_refreshed_data.check_frozen_history`` / ``check_returns_agreement``
  — the workflow guard that rejects a restating pull.
* ``btc_ct_engine`` self-pull — disabled unless explicitly re-enabled.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "scripts"), str(_ROOT / "app"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("yfinance")           # pull script imports it at module level
import pull_backtest_data as pbd          # noqa: E402
import validate_refreshed_data as vrd     # noqa: E402


def _df(n=12, base=100.0, cols=("close", "volume")):
    idx = pd.bdate_range(end="2026-07-31", periods=n)
    return pd.DataFrame({c: base + np.arange(n, dtype=float) + i
                         for i, c in enumerate(cols)}, index=idx)


# ── _freeze_history ──────────────────────────────────────────────────────────
def test_freeze_pins_history_allows_tail_and_append(tmp_path, monkeypatch):
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    prev = _df()
    p = tmp_path / "x.csv"
    prev.to_csv(p)
    fresh = prev * 1.5                                   # full restatement
    extra = pd.DataFrame({"close": [999.0], "volume": [1.0]},
                         index=[prev.index[-1] + pd.Timedelta(days=1)])
    fresh = pd.concat([fresh, extra])
    out = pbd._freeze_history("t", fresh, p, tail=5)
    frozen = prev.index[:-5]
    pd.testing.assert_frame_equal(out.loc[frozen], prev.loc[frozen],
                                  check_freq=False)   # pinned
    tail = prev.index[-5:]
    pd.testing.assert_frame_equal(out.loc[tail], fresh.loc[tail],
                                  check_freq=False)      # tail fresh
    assert float(out["close"].iloc[-1]) == 999.0                       # appended


def test_freeze_restores_lost_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    prev = _df()
    p = tmp_path / "x.csv"
    prev.to_csv(p)
    fresh = prev.iloc[3:] * 2.0                          # lost the oldest 3 rows
    out = pbd._freeze_history("t", fresh, p, tail=5)
    assert prev.index[0] in out.index                    # restored
    pd.testing.assert_frame_equal(out.loc[prev.index[:-5]], prev.iloc[:-5],
                                  check_freq=False)


def test_freeze_tail_zero_fully_pins(tmp_path, monkeypatch):
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    prev = _df()
    p = tmp_path / "x.csv"
    prev.to_csv(p)
    out = pbd._freeze_history("t", prev * 3.0, p, tail=0)
    pd.testing.assert_frame_equal(out, prev, check_freq=False)


def test_freeze_bypass_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULL_UNFROZEN", "1")
    prev = _df()
    p = tmp_path / "x.csv"
    prev.to_csv(p)
    fresh = prev * 1.5
    out = pbd._freeze_history("t", fresh, p, tail=5)
    pd.testing.assert_frame_equal(out, fresh)            # deliberate re-baseline


def test_freeze_no_previous_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PULL_UNFROZEN", raising=False)
    fresh = _df()
    out = pbd._freeze_history("t", fresh, tmp_path / "absent.csv", tail=5)
    pd.testing.assert_frame_equal(out, fresh)


# ── validator: frozen-history agreement ─────────────────────────────────────
def test_validator_accepts_append_and_tail_changes():
    old = _df()
    new = old.copy()
    new.iloc[-2:] *= 1.01                                # tail correction OK
    new.loc[old.index[-1] + pd.Timedelta(days=1)] = [200.0, 1.0]
    assert vrd.check_frozen_history(new, old, tail=5) == []


def test_validator_rejects_restated_history():
    old = _df()
    new = old.copy()
    new.iloc[0, 0] *= 1.10                               # pinned row restated
    errs = vrd.check_frozen_history(new, old, tail=5)
    assert errs and "restated" in errs[0]


def test_validator_rejects_dropped_rows_and_columns():
    old = _df()
    errs = vrd.check_frozen_history(old.iloc[2:], old, tail=5)
    assert any("dropped" in e for e in errs)
    errs = vrd.check_frozen_history(old.drop(columns=["volume"]), old, tail=5)
    assert any("column 'volume' dropped" in e for e in errs)


def test_validator_nan_flip_is_restatement():
    old = _df()
    new = old.copy()
    new.iloc[1, 1] = np.nan
    errs = vrd.check_frozen_history(new, old, tail=5)
    assert errs and "NaN flip" in errs[0]


# ── validator: returns agreement (split-safe) ───────────────────────────────
def test_returns_agreement_tolerates_split_rescale():
    old = _df(n=140)
    new = old * 10.0                                     # 10:1 split re-base
    assert vrd.check_returns_agreement(new, old, "close") == []


def test_returns_agreement_rejects_jittered_history():
    old = _df(n=140)
    new = old.copy()
    new["close"] = new["close"] * (1 + 0.01 * np.sin(np.arange(len(new))))
    errs = vrd.check_returns_agreement(new, old, "close")
    assert errs and "disagree" in errs[0]


# ── BTC engine self-pull kill switch ────────────────────────────────────────
def test_self_pull_disabled_by_default(monkeypatch):
    import btc_ct_engine as bce
    monkeypatch.delenv(bce._SELF_PULL_ENV, raising=False)
    assert not bce._self_pull_allowed()
    monkeypatch.setenv(bce._SELF_PULL_ENV, "1")
    assert bce._self_pull_allowed()
