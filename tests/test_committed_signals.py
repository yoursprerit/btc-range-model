"""Committed signals ignore the in-progress *today* bar.

The Overall engines (``overall_core.run_asset`` and ``gldm_engine.run_gldm``)
consume the quality-gated daily frame, which overlays Yahoo's in-progress
*today* row during US market hours for live prices.  The COMMITTED decision,
however, must come from completed sessions only — exactly like each dedicated
app, which builds its prediction/signal tables on
``freshness.drop_in_progress_us_bar(daily)``.

Regression for the 2026-08-03 GDXM discrepancy: the partial bar's high is only
"the high so far", so scoring it as a completed bar biased err_hi downward and
fired a spurious D2 momentum-fade EXIT on the Overall / Live Signals card while
the Gold Miners app (completed bars only) correctly showed LONG — HOLDING.

No network: the loaders are monkeypatched with the repo's pinned snapshot CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "app"))

import freshness as fr    # noqa: E402
import overall_core as ov  # noqa: E402

CUT = fr.expected_equity_asof()          # last completed US session, real clock


def _load_snapshot(csv: Path, prefix: str | None = None) -> pd.DataFrame:
    if not csv.exists():
        pytest.skip(f"pinned snapshot missing: {csv}")
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    if prefix:
        df = df.rename(columns={c: "px_" + c[len(prefix):]
                                for c in df.columns if c.startswith(prefix)})
    # baseline = completed sessions only (the snapshot should already be, but
    # the real clock decides — never let a stale row past the cutoff in)
    return df[df.index <= CUT]


def _with_partial(df: pd.DataFrame, price_prefix: str) -> pd.DataFrame:
    """Append a distorted in-progress bar one day past the completed cutoff —
    close down 2%, intraday high still BELOW the prior close (the biased-low
    partial high that made D2 fire pre-fix)."""
    last_close = float(df[f"{price_prefix}_close"].dropna().iloc[-1])
    partial = df.iloc[[-1]].copy()
    partial.index = [pd.Timestamp(CUT) + pd.Timedelta(days=1)]
    for suff, mult in (("open", 0.995), ("high", 0.997), ("low", 0.975),
                       ("close", 0.98)):
        col = f"{price_prefix}_{suff}"
        if col in partial.columns:
            partial[col] = last_close * mult
    return pd.concat([df, partial])


def _decisions(results):
    return {r["key"]: (r["decision"]["state"], r["decision"]["label"])
            for r in results}


@pytest.mark.parametrize("key,folder", [("SOXX", "soxx"), ("XLE", "xle")])
def test_run_asset_committed_signal_ignores_partial_bar(monkeypatch, key, folder):
    cfg = ov.overall_config(key)
    base = _load_snapshot(REPO / "data" / folder / "macro_daily.csv",
                          prefix=f"{cfg.primary_symbol.lower()}_")
    if len(base) < 300:
        pytest.skip("snapshot too short")

    monkeypatch.setattr(ov, "_load_daily", lambda _cfg: base.copy())
    res_completed = ov.run_asset(cfg)
    assert res_completed, "engine returned no instruments on the snapshot"

    with_partial = _with_partial(base, "px")
    monkeypatch.setattr(ov, "_load_daily", lambda _cfg: with_partial.copy())
    res_partial = ov.run_asset(cfg)
    assert res_partial

    # the committed decision, signal as-of date and trend line must be
    # identical with and without the half-formed bar …
    assert _decisions(res_partial) == _decisions(res_completed)
    for a, b in zip(res_completed, res_partial):
        assert pd.Timestamp(b["as_of"]) == pd.Timestamp(a["as_of"])
        assert pd.Timestamp(b["as_of"]) <= CUT
        if a["ma_val"] is not None:
            assert b["ma_val"] == pytest.approx(a["ma_val"])
        # … and close_hist stays completed-only: trend_long_now_live APPENDS
        # the live price, so a partial bar left in would count today twice
        assert b["close_hist"][-1] == pytest.approx(a["close_hist"][-1])
    # while the LIVE display price does track the partial bar
    primary = [r for r in res_partial if r["key"] == key][0]
    assert primary["last_close"] == pytest.approx(
        float(with_partial["px_close"].iloc[-1]))


def test_run_gldm_committed_signal_ignores_partial_bar(monkeypatch):
    import gldm_engine as ge

    base = _load_snapshot(REPO / "data" / "gldm" / "gldm_macro_daily.csv")
    if len(base) < 300 or "gldm_close" not in base.columns:
        pytest.skip("gold snapshot unusable")

    monkeypatch.setattr(ge, "_load_daily", lambda: base.copy())
    monkeypatch.setattr(ge, "_attach_nugt", lambda df: df)   # no network
    res_completed = ge.run_gldm()
    assert res_completed

    with_partial = _with_partial(base, "gldm")
    monkeypatch.setattr(ge, "_load_daily", lambda: with_partial.copy())
    res_partial = ge.run_gldm()
    assert res_partial

    assert _decisions(res_partial) == _decisions(res_completed)
    for a, b in zip(res_completed, res_partial):
        assert pd.Timestamp(b["as_of"]) == pd.Timestamp(a["as_of"])
        assert pd.Timestamp(b["as_of"]) <= CUT
