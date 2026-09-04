"""Today's action plan: Price (Close of Last Bar) · Live Price · Chg % agree.

The three columns sit side by side, and the middle one is tinted green/red
against the first — so the third has to be the move between exactly those two
numbers.  It wasn't: it printed the spot feed's SESSION day-change (live vs the
*previous calendar session's* close), which measures a different interval and
contradicted the two prices in nearly every row.  Observed on 2026-09-04, a few
minutes after the 4 PM ET close, with every equity sleeve's app having ingested
today's bar:

    key    last bar close   live price   Chg % shown   live vs bar
    SOXL         117.54       117.54       +10.12%         0.00%
    NUGT         191.94       191.94        -4.30%         0.00%
    GDX           99.29        99.29        -2.17%         0.00%
    MSTR         142.07       142.80        -1.39%        +0.51%
    ETH         2525.55      2455.16        -2.11%        -2.79%

The equity rows printed the same price twice next to a double-digit change; the
BTC-app sleeves (12:00-UTC bar anchor / cached vintage) were measured against a
close the row never displays.  Both now reconcile: ``Chg %`` is
``live_change_pct(last_close, live_price)``, and the session change — still a
real number from the feed — rides along as a sub-line when it differs.

Also pinned here: a row with NO live quote reads "—" rather than a fabricated
0.00% (the overlay used to copy the bar close into ``live_price`` for a key
whose fetch failed, making the two states indistinguishable), and the SATA row
shows the quote's previous close in the last-bar column instead of its $100 par
cost basis, so its Chg % is the same live-vs-last-bar move as every row above.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402

_APP = (Path(__file__).resolve().parent.parent / "app" / "overall_app.py").read_text()
_PLAN_BLOCK = _APP.split("# ── 2. TODAY'S ACTION PLAN")[1].split(
    "# ── 3. PER-SIGNAL LIVE CARDS")[0]
_SPOT_BLOCK = _APP.split("# ── live spot prices")[1].split("# ── small HTML helpers")[0]


# ── the calculation itself ───────────────────────────────────────────────
def test_chg_is_the_move_between_the_two_price_columns():
    assert oc.live_change_pct(100.0, 101.5) == pytest.approx(1.5)
    assert oc.live_change_pct(2525.55, 2455.16) == pytest.approx(-2.7871, abs=1e-4)
    # the row that used to print two identical prices beside +10.12%
    assert oc.live_change_pct(117.54, 117.54) == pytest.approx(0.0)


def test_chg_is_none_without_a_usable_pair():
    assert oc.live_change_pct(100.0, None) is None      # no live quote → "—"
    assert oc.live_change_pct(None, 100.0) is None
    assert oc.live_change_pct(0.0, 100.0) is None       # no divide-by-zero row
    assert oc.live_change_pct(float("nan"), 100.0) is None
    assert oc.live_change_pct(100.0, float("nan")) is None
    assert oc.live_change_pct("n/a", 100.0) is None


def test_observed_rows_reconcile_with_the_prices_displayed():
    """Every 2026-09-04 row above: the column now equals the printed move."""
    observed = [  # (last bar close, live price, session day-change the feed gave)
        (117.54, 117.54, 10.12), (191.94, 191.94, -4.30), (99.29, 99.29, -2.17),
        (142.07, 142.80, -1.39), (2525.55, 2455.16, -2.11),
    ]
    for bar_px, live_px, session in observed:
        chg = oc.live_change_pct(bar_px, live_px)
        assert chg == pytest.approx((live_px / bar_px - 1) * 100)
        # the point of the fix: the session change was NOT this move
        assert abs(session - chg) > 0.05


# ── the overlay that feeds the columns ───────────────────────────────────
def test_live_price_is_only_set_from_a_real_quote():
    # apply_spot leaves r["last_close"] on the bar close when a key's fetch
    # failed, so live_price must come off the quote dict, guarded on its price
    assert '_q = _spot.get(_a["key"]) or {}' in _SPOT_BLOCK
    assert 'if _q.get("price"):' in _SPOT_BLOCK
    assert '_a["live_price"] = float(_q["price"])' in _SPOT_BLOCK
    assert '_a["live_price"] = _r["last_close"]' not in _SPOT_BLOCK


def test_apply_spot_leaves_the_bar_close_when_a_quote_is_missing():
    res = [dict(key="AAA", last_close=10.0, dchg=1.0, pos={}),
           dict(key="BBB", last_close=20.0, dchg=2.0, pos={})]
    oc.apply_spot(res, {"AAA": {"price": 11.0, "dchg": 5.0},
                        "BBB": {"price": None, "dchg": None}})
    assert res[0]["last_close"] == 11.0            # overlaid with the live quote
    assert res[1]["last_close"] == 20.0            # untouched — no quote came back


# ── the rendered cell ────────────────────────────────────────────────────
def test_chg_cell_uses_the_two_displayed_prices_not_the_session_change():
    assert "_chg = ov.live_change_pct(_bar_px, _live_px if _has_live else None)" \
        in _PLAN_BLOCK
    assert 'chg_s = f"{_chg:+.2f}%"' in _PLAN_BLOCK
    # the old source of the number is gone from the cell
    assert '_dchg = _r.get("dchg")' not in _PLAN_BLOCK
    # …and the session change survives only as the labelled sub-line
    assert "session {_sess:+.2f}%" in _PLAN_BLOCK


def test_session_subline_is_rendered_for_a_zero_move_too():
    """The 0.00% rows are exactly the ones the old column got wrong, so the
    session sub-line must not hang off the non-zero branch."""
    body = _PLAN_BLOCK.split("_chg = ov.live_change_pct")[1].split("# cost basis")[0]
    assert "if _chg is not None:" in body
    zero_branch = body.split("elif abs(_chg) < 0.005:")[1].split("else:")[0]
    assert 'chg_s, chg_col = "0.00%"' in zero_branch      # never a red "-0.00%"
    assert "_sess" not in zero_branch                     # sub-line applies here too


def test_missing_quote_renders_an_em_dash_not_a_zero():
    assert '_has_live = _live_px is not None' in _PLAN_BLOCK
    assert 'if _chg is None:\n                chg_s, chg_col = "—", "#94a3b8"' \
        in _PLAN_BLOCK


def test_live_price_tint_and_chg_share_one_baseline():
    """The green/red tint on Live Price and Chg % must compare to the SAME
    number — the last bar's close — or the colour can contradict the sign."""
    assert "_bar_px = a[\"last_close\"]" in _PLAN_BLOCK
    assert "_live_px_col = (C_BUY if _live_px > _bar_px" in _PLAN_BLOCK
    assert "else C_EXIT if _live_px < _bar_px else \"inherit\")" in _PLAN_BLOCK


# ── the quote feed behind the session sub-line ───────────────────────────
def _fake_chart(bar_days, closes, quote_epoch, price, tz="America/New_York"):
    """A Yahoo v8 chart payload: daily bars at 13:30 UTC (US session start)."""
    import pandas as pd

    ts = [int(pd.Timestamp(f"{d} 13:30", tz="UTC").timestamp()) for d in bar_days]

    class _R:
        status_code = 200

        def json(self):
            return {"chart": {"result": [{
                "meta": {"regularMarketPrice": price, "regularMarketTime": quote_epoch,
                         "exchangeTimezoneName": tz},
                "timestamp": ts,
                "indicators": {"quote": [{"close": closes}]}}]}}

    return _R()


def test_quote_prev_is_the_close_before_the_quotes_own_session(monkeypatch):
    """The chart can lag the quote early in a session — today's bar missing
    while regularMarketPrice is already live. Picking s.iloc[-2] positionally
    then measured against the session BEFORE last, overstating the change by a
    whole day; the previous close is chosen by the quote's own date instead."""
    import pandas as pd

    quote_epoch = int(pd.Timestamp("2026-09-04 18:00", tz="UTC").timestamp())
    monkeypatch.setattr(
        oc.requests, "get",
        lambda *a, **k: _fake_chart(["2026-09-02", "2026-09-03"], [100.0, 102.0],
                                    quote_epoch, 105.0))
    px, prev = oc._quote("TEST")
    assert px == 105.0
    assert prev == 102.0                       # yesterday's close, not 100.0
    assert oc.live_change_pct(prev, px) == pytest.approx(2.9412, abs=1e-4)


def test_quote_prev_skips_the_in_progress_bar(monkeypatch):
    """The ordinary case — today's (partial) bar is in the series, so the
    previous close is the one before it."""
    import pandas as pd

    quote_epoch = int(pd.Timestamp("2026-09-04 18:00", tz="UTC").timestamp())
    monkeypatch.setattr(
        oc.requests, "get",
        lambda *a, **k: _fake_chart(["2026-09-02", "2026-09-03", "2026-09-04"],
                                    [100.0, 102.0, 105.0], quote_epoch, 105.0))
    px, prev = oc._quote("TEST")
    assert (px, prev) == (105.0, 102.0)


# ── the SATA row follows the same rule ───────────────────────────────────
def test_fetch_sata_reports_the_previous_close(monkeypatch):
    monkeypatch.setattr(oc, "_quote", lambda sym: (101.0, 100.5))
    q = oc.fetch_sata()
    assert q["price"] == 101.0 and q["prev"] == 100.5
    assert oc.live_change_pct(q["prev"], q["price"]) == pytest.approx(q["dchg"])


def test_sata_row_shows_a_bar_close_not_its_par_basis():
    assert "_sa_prev = _sata.get(\"prev\")" in _PLAN_BLOCK
    assert "_sa_dc = ov.live_change_pct(_sa_prev, _sa_px)" in _PLAN_BLOCK
    # the last-bar cell renders the previous close; par stays on the P&L sub-line
    assert "tabular-nums'>{sa_prev_s}</td>" in _PLAN_BLOCK
    assert "tabular-nums'>${si['par']:,.2f}</td>" not in _PLAN_BLOCK
    assert "@ $100.00 par" in _PLAN_BLOCK


# ── the caption has to say what the column is ────────────────────────────
def test_caption_documents_the_chg_column():
    assert "**Chg %** is exactly the move between those two " in _PLAN_BLOCK
    assert "live vs last bar" in _PLAN_BLOCK          # the header's sub-label
