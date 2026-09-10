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
``live_change_pct(last_close, live_price)`` — ONE value in the column, the move
from the price on its left to the price on its right, for that instrument's own
bar.  (The session day-change briefly rode along as a sub-line; a second figure
in the same cell only relocated the ambiguity, so the column carries the
bar-relative move alone.)

Also pinned here: a row with NO live quote reads "—" rather than a fabricated
0.00% (the overlay used to copy the bar close into ``live_price`` for a key
whose fetch failed, making the two states indistinguishable); the SATA row
shows the quote's previous close in the last-bar column instead of its $100 par
cost basis, so its Chg % is the same live-vs-last-bar move as every row above;
and every live price resolves through a primary→backup provider chain, so one
feed going down empties neither the Live Price nor the Chg % column.
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


def test_chg_cell_shows_exactly_one_value():
    """One number in the column, and it is the bar-relative move. A second
    figure in the cell (the session day-change, once carried as a sub-line)
    only relocated the ambiguity the column was fixed to remove."""
    cell = _PLAN_BLOCK.split("_chg = ov.live_change_pct")[1].split("# cost basis")[0]
    assert "_sess" not in cell and "session {" not in cell
    # exactly one figure can reach the cell: the em-dash, 0.00%, or ±x.xx%
    assert sorted(l.strip() for l in cell.splitlines()
                  if "chg_s = " in l or "chg_s, chg_col = " in l) == [
        'chg_s = f"{_chg:+.2f}%"',
        'chg_s, chg_col = "0.00%", "#64748b"',
        'chg_s, chg_col = "—", "#94a3b8"',
    ]
    # the only other thing appended is a WORD — the verification label, which
    # carries no number and so cannot be mistaken for a second measurement
    appended = [l for l in cell.splitlines() if "chg_s +=" in l]
    assert appended and not any("%" in l and "{" in l for l in appended)


def test_zero_move_is_neutral_not_a_red_minus_zero():
    body = _PLAN_BLOCK.split("_chg = ov.live_change_pct")[1].split("# cost basis")[0]
    zero_branch = body.split("elif abs(_chg) < 0.005:")[1].split("else:")[0]
    assert 'chg_s, chg_col = "0.00%"' in zero_branch


def test_missing_quote_renders_an_em_dash_not_a_zero():
    assert '_has_live = _live_px is not None' in _PLAN_BLOCK
    assert 'if _chg is None:\n                chg_s, chg_col = "—", "#94a3b8"' \
        in _PLAN_BLOCK


def test_live_price_tint_and_chg_share_one_baseline():
    """The green/red tint on Live Price and Chg % must compare to the SAME
    number — the last bar's close — or the colour can contradict the sign."""
    assert '_bar_px = a.get("bar_close")' in _PLAN_BLOCK
    assert "_live_px_col = (C_BUY if _live_px > _bar_px" in _PLAN_BLOCK
    assert "else C_EXIT if _live_px < _bar_px else \"inherit\")" in _PLAN_BLOCK


# ── the quote feed behind the live prices ────────────────────────────────
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
    q = oc._quote("TEST")
    assert q["price"] == 105.0
    assert q["prev"] == 102.0                  # yesterday's close, not 100.0
    assert oc.live_change_pct(q["prev"], q["price"]) == pytest.approx(2.9412, abs=1e-4)
    # …and the same one request carries what the basis check needs
    assert q["bars"]["2026-09-03"] == 102.0
    assert q["quote_time"] == pd.Timestamp(quote_epoch, unit="s", tz="UTC")


def test_quote_prev_skips_the_in_progress_bar(monkeypatch):
    """The ordinary case — today's (partial) bar is in the series, so the
    previous close is the one before it."""
    import pandas as pd

    quote_epoch = int(pd.Timestamp("2026-09-04 18:00", tz="UTC").timestamp())
    monkeypatch.setattr(
        oc.requests, "get",
        lambda *a, **k: _fake_chart(["2026-09-02", "2026-09-03", "2026-09-04"],
                                    [100.0, 102.0, 105.0], quote_epoch, 105.0))
    q = oc._quote("TEST")
    assert (q["price"], q["prev"]) == (105.0, 102.0)
    assert set(q["bars"]) == {"2026-09-02", "2026-09-03", "2026-09-04"}


# ── the SATA row follows the same rule ───────────────────────────────────
def test_fetch_sata_reports_the_previous_close(monkeypatch):
    monkeypatch.setattr(oc, "_quote", lambda sym: dict(price=101.0, prev=100.5))
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
    assert "**Chg %** is one number and one only: " in _PLAN_BLOCK
    assert "Live Price vs Close of Last Bar" in _PLAN_BLOCK   # header sub-label


# ════════════════════════════════════════════════════════════════════════════
# BACKUP LIVE-PRICE FEED
# ════════════════════════════════════════════════════════════════════════════
# Yahoo is the only live-quote source the cockpit had. When it fails for a
# symbol — rate-limited or 403'd on a shared egress IP, geo-blocked, or a null
# regularMarketPrice — that row's Live Price froze on its last completed bar and
# Chg % read "—", exactly when the market was moving. Each quote now falls back,
# per symbol, to an independent keyless provider: Nasdaq for US listed names,
# Coinbase then Binance for spot crypto.
import market_fallback as mf  # noqa: E402


def test_backup_chain_order_is_primary_first():
    assert oc.PRIMARY_QUOTE_SRC == "yahoo"
    assert [n for n, _ in mf._CHAINS] == ["nasdaq", "coinbase", "binance"]


def test_quote_prefers_yahoo_and_never_calls_a_backup_when_it_answers(monkeypatch):
    called = []
    monkeypatch.setattr(oc, "_quote", lambda sym: dict(price=101.0, prev=100.0))
    monkeypatch.setattr(mf, "live_quote", lambda sym: called.append(sym) or (1.0, 1.0, "x"))
    assert oc.quote("SOXL") == dict(price=101.0, prev=100.0, src="yahoo")
    assert called == []                       # backups are untouched while Yahoo works


def test_quote_falls_back_when_yahoo_returns_nothing(monkeypatch):
    monkeypatch.setattr(oc, "_quote", lambda sym: {})
    monkeypatch.setattr(mf, "live_quote", lambda sym: (124.82, 123.27, "nasdaq"))
    q = oc.quote("SOXL")
    assert (q["price"], q["prev"], q["src"]) == (124.82, 123.27, "nasdaq")
    assert q["bars"] == {} and q["quote_time"] is None


def test_quote_falls_back_when_yahoo_raises(monkeypatch):
    def _boom(sym):
        raise RuntimeError("403 from the shared egress IP")
    monkeypatch.setattr(oc, "_quote", _boom)
    monkeypatch.setattr(mf, "live_quote", lambda sym: (78421.86, 78900.0, "coinbase"))
    q = oc.quote("BTC-USD")
    assert (q["price"], q["prev"], q["src"]) == (78421.86, 78900.0, "coinbase")


def test_quote_reports_nothing_when_every_feed_is_down(monkeypatch):
    monkeypatch.setattr(oc, "_quote", lambda sym: {})
    monkeypatch.setattr(mf, "live_quote", lambda sym: (None, None, None))
    assert oc.quote("SOXL") == {}


def test_failover_is_per_symbol_not_universe_wide(monkeypatch):
    """One name Yahoo won't quote must not push the others onto a backup."""
    monkeypatch.setattr(oc, "_quote",
                        lambda sym: {} if sym == "SOXL" else dict(price=10.0, prev=9.0))
    monkeypatch.setattr(mf, "live_quote", lambda sym: (11.0, 10.5, "nasdaq"))
    spot = oc.fetch_spot({"SOXL": "SOXL", "XLE": "XLE"})
    assert (spot["SOXL"]["price"], spot["SOXL"]["prev"], spot["SOXL"]["src"]) == \
        (11.0, 10.5, "nasdaq")
    assert spot["SOXL"]["dchg"] == pytest.approx(4.7619, abs=1e-4)
    assert spot["XLE"]["src"] == "yahoo" and spot["XLE"]["price"] == 10.0
    assert oc.backup_quote_keys(spot) == [("SOXL", "nasdaq")]


def test_backup_quote_keys_is_empty_on_the_normal_path():
    assert oc.backup_quote_keys({"XLE": dict(price=10.0, src="yahoo"),
                                 "SOXL": dict(price=None, src=None)}) == []


def test_sata_quote_also_falls_back(monkeypatch):
    monkeypatch.setattr(oc, "_quote", lambda sym: {})
    monkeypatch.setattr(mf, "live_quote", lambda sym: (99.97, 99.84, "nasdaq"))
    q = oc.fetch_sata()
    assert q["price"] == 99.97 and q["prev"] == 99.84 and q["src"] == "nasdaq"


# ── the providers themselves (parsing, not the network) ──────────────────
def _json_patch(monkeypatch, payloads):
    """Serve canned JSON per URL substring; None → that provider fails."""
    def _fake(url, headers=None, timeout=None):
        for frag, payload in payloads.items():
            if frag in url:
                return payload
        return None
    monkeypatch.setattr(mf, "_get_json", _fake)


def test_nasdaq_quote_parses_price_and_derives_the_previous_close(monkeypatch):
    _json_patch(monkeypatch, {"MSTR": {"data": {"primaryData": {
        "lastSalePrice": "$134.485", "netChange": "-2.035",
        "percentageChange": "-1.49%"}}}})
    px, prev = mf.nasdaq_quote("MSTR")
    assert px == pytest.approx(134.485)
    assert prev == pytest.approx(136.52)                 # last − netChange
    assert oc.live_change_pct(prev, px) == pytest.approx(-1.49, abs=0.01)


def test_nasdaq_quote_skips_symbols_it_cannot_serve():
    for sym in ("^VIX", "GC=F", "DX-Y.NYB", "BTC-USD"):
        assert mf.nasdaq_quote(sym) == (None, None)


def test_nasdaq_quote_is_none_on_a_junk_payload(monkeypatch):
    _json_patch(monkeypatch, {"SOXL": {"data": {"primaryData": {
        "lastSalePrice": "N/A", "netChange": "N/A"}}}})
    assert mf.nasdaq_quote("SOXL") == (None, None)


def test_crypto_providers_parse_their_own_shapes(monkeypatch):
    _json_patch(monkeypatch, {
        "coinbase": {"open": "78900", "last": "78605.14"},
        "binance": {"lastPrice": "78600.67", "prevClosePrice": "78549.53"}})
    assert mf.coinbase_quote("BTC-USD") == (pytest.approx(78605.14),
                                            pytest.approx(78900.0))
    assert mf.binance_quote("BTC-USD") == (pytest.approx(78600.67),
                                           pytest.approx(78549.53))
    assert mf.coinbase_quote("SOXL") == (None, None)     # equities aren't crypto


def test_live_quote_walks_the_chain_to_the_first_feed_that_answers(monkeypatch):
    _json_patch(monkeypatch, {"binance": {"lastPrice": "78600.67",
                                          "prevClosePrice": "78549.53"}})
    # nasdaq refuses the symbol, coinbase returns nothing → binance serves it
    px, prev, src = mf.live_quote("BTC-USD")
    assert src == "binance" and px == pytest.approx(78600.67)


def test_live_quote_survives_a_provider_that_raises(monkeypatch):
    monkeypatch.setattr(mf, "nasdaq_quote", lambda s: (_ for _ in ()).throw(OSError("down")))
    _json_patch(monkeypatch, {"coinbase": {"open": "2498.35", "last": "2490.01"}})
    assert mf.live_quote("ETH-USD") == (pytest.approx(2490.01),
                                        pytest.approx(2498.35), "coinbase")


# ── the cockpit says which prices are not the primary feed's ─────────────
def test_a_backup_price_is_labelled_in_the_table():
    assert '_q = _spot.get(a["key"]) or {}' in _PLAN_BLOCK
    assert '_src = _q.get("src")' in _PLAN_BLOCK
    assert 'if _has_live and _src and _src != ov.PRIMARY_QUOTE_SRC:' in _PLAN_BLOCK
    assert "via {_src}" in _PLAN_BLOCK
    assert "via {_sa_src}" in _PLAN_BLOCK                # …and on the SATA row


def test_the_price_banner_names_backup_served_instruments():
    banner = _APP.split("_px_note = ")[0].split("_auto = ")[1]
    assert "_bk = ov.backup_quote_keys(_spot)" in banner
    assert "primary quote feed unavailable for" in banner


# ════════════════════════════════════════════════════════════════════════════
# BOTH INPUTS ARE CHECKED BEFORE Chg % IS COMPUTED
# ════════════════════════════════════════════════════════════════════════════
# The column was differencing two prices nobody had validated. The basis came
# from ``last_close``, which is the DISPLAY price — and ``daily`` deliberately
# keeps the in-progress US bar, so during market hours it is today's running
# price, not a close. Measured live on 2026-09-10 at 14:13 ET with the session
# open, every equity sleeve showed today's live price in the "Close of Last Bar"
# column and Chg % therefore read 0.00% all session:
#
#     key    shown as "last bar"   real Sep 9 close   error
#     SOXL          117.14              125.87        -6.9%
#     NUGT          180.07              192.46        -6.4%
#     REMX           72.31               76.34        -5.3%
#     GDX            96.29               99.47        -3.2%
#     SOXX          519.50              532.00        -2.4%
#
# The basis is now each engine's ``bar_close`` — the traded instrument's own
# last COMPLETED session close, carrying ``bar_date`` — and both it and the live
# quote go through ``verify_price_pair`` first.
import freshness as frs  # noqa: E402


def _bars(**closes):
    return {"price": 100.0, "bars": dict(closes),
            "quote_time": pd.Timestamp.utcnow(), "session_closed": False}


import pandas as pd  # noqa: E402


# ── the completed-bar basis ──────────────────────────────────────────────
def test_completed_bar_takes_the_last_finite_close_and_its_date():
    df = pd.DataFrame({"px_close": [10.0, 11.0, None, 12.0]},
                      index=pd.to_datetime(["2026-09-04", "2026-09-08",
                                            "2026-09-09", "2026-09-10"]))
    assert frs.completed_bar(df, "px_close") == (12.0, pd.Timestamp("2026-09-10"))


def test_completed_bar_reports_nothing_rather_than_guessing():
    empty = pd.DataFrame({"px_close": []})
    assert frs.completed_bar(empty, "px_close")[1] is None
    assert frs.completed_bar(pd.DataFrame({"a": [1.0]}), "px_close")[1] is None
    zeros = pd.DataFrame({"px_close": [0.0, -1.0]},
                         index=pd.to_datetime(["2026-09-09", "2026-09-10"]))
    assert zeros.pipe(frs.completed_bar, "px_close")[1] is None


def test_the_in_progress_bar_is_what_completed_bar_excludes():
    """The Sep 10 defect in one assertion: the frame the display price comes
    from carries today's running price; the frame the basis comes from does
    not."""
    idx = pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10"])
    daily = pd.DataFrame({"px_close": [123.27, 125.87, 117.14]}, index=idx)   # SOXL
    hist = frs.drop_in_progress_us_bar(
        daily, now=pd.Timestamp("2026-09-10 18:13", tz="UTC"))
    assert frs.completed_bar(daily, "px_close")[0] == 117.14      # in-progress
    assert frs.completed_bar(hist, "px_close") == (125.87, pd.Timestamp("2026-09-09"))


# ── the verifier ─────────────────────────────────────────────────────────
def test_a_reconciled_pair_passes_clean():
    v = oc.verify_price_pair(125.87, "2026-09-09", _bars(**{"2026-09-09": 125.87}))
    assert v["ok"] and v["basis_ok"] and v["quote_ok"] and v["flags"] == []
    assert v["official"] == pytest.approx(125.87)


def test_a_basis_the_tape_disagrees_with_is_flagged_not_silently_differenced():
    """The in-progress print, caught: 117.14 is not what Sep 9 closed at."""
    v = oc.verify_price_pair(117.14, "2026-09-09", _bars(**{"2026-09-09": 125.87}))
    assert not v["basis_ok"] and not v["ok"]
    assert "basis_mismatch" in v["flags"]
    assert "125.87" in v["notes"][0]


def test_basis_tolerance_admits_a_rounding_gap_only():
    on_edge = 125.87 * (1 + oc.BASIS_TOL_PCT / 100 * 0.9)
    over = 125.87 * (1 + oc.BASIS_TOL_PCT / 100 * 1.5)
    assert oc.verify_price_pair(on_edge, "2026-09-09",
                                _bars(**{"2026-09-09": 125.87}))["basis_ok"]
    assert not oc.verify_price_pair(over, "2026-09-09",
                                    _bars(**{"2026-09-09": 125.87}))["basis_ok"]


def test_a_bar_behind_the_latest_close_says_which_bar_it_is():
    v = oc.verify_price_pair(125.87, "2026-09-03",
                             _bars(**{"2026-09-09": 130.0}),
                             expected_bar="2026-09-09")
    assert "bar_behind" in v["flags"]
    assert "Sep 3" in v["notes"][-1] and "Sep 9" in v["notes"][-1]


def test_a_12utc_bar_is_dated_but_not_cross_checked():
    """The CT engine's bar and the daily feed's close for the same date are
    different bars, so comparing them would manufacture a mismatch daily."""
    q = _bars(**{"2026-09-09": 79500.0})            # UTC-midnight close
    v = oc.verify_price_pair(77882.31, "2026-09-09", q, cross_check=False)
    assert v["ok"] and v["basis_ok"] and v["flags"] == []
    assert oc.verify_price_pair(77882.31, "2026-09-09", q)["flags"] == ["basis_mismatch"]


def test_an_absent_basis_fails_closed():
    for bad in (None, float("nan"), 0.0, "n/a"):
        v = oc.verify_price_pair(bad, "2026-09-09", _bars())
        assert not v["ok"] and v["flags"] == ["no_basis"]


def test_an_undated_basis_cannot_be_verified():
    v = oc.verify_price_pair(125.87, None, _bars(**{"2026-09-09": 125.87}))
    assert not v["basis_ok"] and "undated_basis" in v["flags"]


# ── the live half ────────────────────────────────────────────────────────
def test_a_quote_that_stopped_ticking_mid_session_is_not_live():
    now = pd.Timestamp("2026-09-10 18:13", tz="UTC")
    q = {"price": 117.30, "bars": {"2026-09-09": 125.87},
         "quote_time": now - pd.Timedelta(hours=3), "session_closed": False}
    v = oc.verify_price_pair(125.87, "2026-09-09", q, now=now)
    assert not v["quote_ok"] and "stale_quote" in v["flags"]
    assert v["age_mins"] == pytest.approx(180.0)


def test_the_last_print_after_the_close_is_not_stale():
    """Outside the session the newest regular print is hours old by design —
    that is the close, not a frozen tick."""
    now = pd.Timestamp("2026-09-10 23:00", tz="UTC")
    q = {"price": 117.30, "bars": {"2026-09-09": 125.87},
         "quote_time": now - pd.Timedelta(hours=3), "session_closed": True}
    assert oc.verify_price_pair(125.87, "2026-09-09", q, now=now)["ok"]


def test_a_missing_quote_fails_the_live_half():
    v = oc.verify_price_pair(125.87, "2026-09-09",
                             {"price": None, "bars": {"2026-09-09": 125.87}})
    assert not v["quote_ok"] and "no_quote" in v["flags"]


def test_a_backup_feed_quote_is_checked_for_freshness_not_reconciled():
    """Backups carry no bar history or session clock, so the basis is reported
    unverified rather than assumed good — and never called wrong."""
    q = {"price": 117.30, "bars": {}, "quote_time": None,
         "session_closed": None, "src": "nasdaq"}
    v = oc.verify_price_pair(125.87, "2026-09-09", q)
    assert v["quote_ok"] and "basis_unverified" in v["flags"]
    assert "basis_mismatch" not in v["flags"]


# ── the engines report a real completed close ────────────────────────────
_CORE = (Path(__file__).resolve().parent.parent / "app" / "overall_core.py").read_text()
_GLDM = (Path(__file__).resolve().parent.parent / "app" / "gldm_engine.py").read_text()
_CT = (Path(__file__).resolve().parent.parent / "app" / "btc_ct_engine.py").read_text()


def test_every_engine_reports_bar_close_off_the_completed_frame():
    # ticker + gold: the in-progress-dropped frame, never `daily`
    assert "bar_close, bar_date = _frs.completed_bar(\n        (hist if hist is not None else daily), col)" in _CORE
    assert "bar_close, bar_date = _frs.completed_bar(hist, col)" in _GLDM
    # CT: equity sleeves off their own series with the partial bar dropped
    assert "_frs.drop_in_progress_us_bar(s.to_frame(\"close\"))" in _CT
    assert 'if key in ("MSTR", "MSTU"):' in _CT
    for src in (_CORE, _GLDM, _CT):
        assert "bar_close=bar_close, bar_date=bar_date," in src


def test_the_gate_carries_the_verified_basis_to_the_action_rows():
    assert 'bar_close=res.get("bar_close"),' in _CORE
    assert 'bar_date=res.get("bar_date"),' in _CORE
    assert 'bar_anchor=res.get("bar_anchor", "session"),' in _CORE


def test_the_published_payload_is_unaffected_by_the_new_fields():
    """bar_date is a Timestamp; the book trims actions to a fixed field set, so
    it can never reach the JSON artifact."""
    import target_book as tb
    payload = tb.build_payload(
        as_of="2026-09-09", profile="balanced", weights={"XLE": 1.0},
        cash_weight=0.0, exec_price={"XLE": 65.31},
        actions=[dict(key="XLE", action="HOLD", decision="LONG", target=1.0,
                      in_pos=True, priority=0.7, exits_next_bar=False,
                      bar_close=65.31, bar_date=pd.Timestamp("2026-09-09"),
                      bar_anchor="session")])
    import json
    json.dumps(payload)                      # would raise on a Timestamp
    assert set(payload["actions"][0]) == {"key", "action", "decision", "target",
                                          "in_pos", "priority", "exits_next_bar"}


# ── the row renders the verified pair ────────────────────────────────────
def test_the_row_measures_from_bar_close_not_the_display_price():
    assert '_bar_px = a.get("bar_close")' in _PLAN_BLOCK
    assert '_bar_dt = a.get("bar_date")' in _PLAN_BLOCK
    assert "_vfy = ov.verify_price_pair(" in _PLAN_BLOCK
    # the display price survives only as the last-resort fallback
    assert '_bar_px, _bar_dt = a["last_close"], None' in _PLAN_BLOCK


def test_a_failed_quote_check_suppresses_the_number():
    assert '_has_live = _live_px is not None and _vfy["quote_ok"]' in _PLAN_BLOCK


def test_the_basis_cell_shows_the_session_it_closed_and_any_complaint():
    assert "{pd.Timestamp(_bar_dt):%b %-d} close" in _PLAN_BLOCK
    assert "⚠️ {_basis_note}" in _PLAN_BLOCK
    assert 'unverified basis' in _PLAN_BLOCK


def test_the_12utc_sleeves_are_not_cross_checked_in_the_row():
    assert 'cross_check=(_anchor == "session")' in _PLAN_BLOCK
    assert '_expected_bar(_anchor)' in _PLAN_BLOCK
