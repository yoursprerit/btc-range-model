"""The SATA idle-cash leg: coupon by default, SATA's real total return where
its daily history exists (``data/sata/sata_daily.csv``)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import overall_core as oc  # noqa: E402
import health_core as hc  # noqa: E402

LIST = pd.Timestamp("2026-01-05")                       # a Monday


def _hist(tmp_path):
    idx = pd.bdate_range(LIST, periods=5)
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": [100.0, 99.0, 99.0, 101.0, 100.0],
                       "volume": 1000.0, "dividend": [0, 0, 1.0, 0, 0]}, index=idx)
    df.index.name = "date"
    p = tmp_path / "sata_daily.csv"
    df.to_csv(p)
    return p, idx


def test_load_sata_returns_is_close_plus_dividend_over_previous_close(tmp_path):
    p, idx = _hist(tmp_path)
    tr = oc.load_sata_returns(p)
    assert list(tr.index) == list(idx) and tr.loc[idx[0]] == 0.0   # listing session
    assert tr.loc[idx[1]] == pytest.approx(99 / 100 - 1)
    assert tr.loc[idx[2]] == pytest.approx((99 + 1.0) / 99 - 1)      # dividend counts
    assert oc.load_sata_returns(tmp_path / "missing.csv") is None


def test_sata_leg_float_is_the_coupon_on_business_days():
    idx = pd.date_range("2026-01-02", "2026-01-06")               # Fri … Tue
    r = oc.sata_leg(0.001, idx)
    assert r.tolist() == [0.001, 0.0, 0.0, 0.001, 0.001]
    assert oc.sata_leg(0.001, idx, business_days_only=False).tolist() == [0.001] * 5


def test_sata_leg_series_real_after_listing_coupon_before_zero_off_session(tmp_path):
    p, idx = _hist(tmp_path)
    tr = oc.load_sata_returns(p)
    cal = pd.date_range(LIST - pd.Timedelta(days=4), idx[-1] + pd.Timedelta(days=2))
    r = pd.Series(oc.sata_leg(tr, cal), index=cal)
    assert r.loc[LIST - pd.Timedelta(days=4)] == pytest.approx(oc.SATA_DAILY)   # Thu before: coupon
    assert r.loc[LIST - pd.Timedelta(days=2)] == 0.0                            # Sat before: nothing
    assert r.loc[LIST] == 0.0                   # listing session: real record starts, no return yet
    assert r.loc[idx[1]] == pytest.approx(-0.01)
    assert r.loc[idx[-1] + pd.Timedelta(days=1)] == 0.0        # Saturday after listing
    assert r.loc[idx[-1] + pd.Timedelta(days=2)] == 0.0


def test_sata_scalar_collapses_a_series_to_the_coupon(tmp_path):
    p, _ = _hist(tmp_path)
    assert oc._sata_scalar(oc.load_sata_returns(p)) == oc.SATA_DAILY
    assert oc._sata_scalar(0.002) == 0.002


def _book(idx):
    rets = pd.DataFrame({"AAA": 0.0}, index=idx)
    w = pd.DataFrame({"AAA": 0.0}, index=idx)
    sata = pd.Series(1.0, index=idx)                      # all idle, all in SATA
    return rets, w, sata


def test_daily_replay_and_attribution_earn_the_real_sata_return(tmp_path):
    p, idx = _hist(tmp_path)
    tr = oc.load_sata_returns(p)
    rets, w, sata = _book(idx)
    d = oc.pnl_daily_replay(rets, w, sata, idx[0], sata_daily=tr)
    a = oc.pnl_attribution_replay(rets, w, sata, idx[0], sata_daily=tr)
    assert d["ret"].loc[idx[1]] == pytest.approx(-0.01)
    assert d["ret"].loc[idx[2]] == pytest.approx((99 + 1.0) / 99 - 1)
    assert a is not None and "sata" in a
    # the coupon path is untouched
    c = oc.pnl_daily_replay(rets, w, sata, idx[0], sata_daily=oc.SATA_DAILY)
    assert np.allclose(c["ret"].iloc[1:], oc.SATA_DAILY)


def test_realized_book_returns_use_the_series(tmp_path):
    p, idx = _hist(tmp_path)
    tr = oc.load_sata_returns(p)
    books = [dict(as_of=str((idx[0] - pd.Timedelta(days=1)).date()), weights={},
                  cash_weight=1.0)]
    rets = pd.DataFrame({"AAA": 0.0}, index=idx)
    live = hc.realized_book_returns(books, rets, tr)
    assert live.loc[idx[1]] == pytest.approx(-0.01)
    coup = hc.realized_book_returns(books, rets, oc.SATA_DAILY)
    assert coup.loc[idx[1]] == pytest.approx(oc.SATA_DAILY)


def test_sata_slice_metrics_real_series_has_a_drawdown(tmp_path):
    p, idx = _hist(tmp_path)
    tr = oc.load_sata_returns(p)
    m = oc.sata_slice_metrics(idx, idx[0], sata_daily=tr)
    assert m["mdd"] == pytest.approx(-0.01)
    assert m["total_ret"] == pytest.approx((1 + tr).prod() - 1)
    flat = oc.sata_slice_metrics(idx, idx[0])
    assert flat["mdd"] == 0.0 and flat["total_ret"] == pytest.approx((1 + oc.SATA_DAILY) ** 4 - 1)
