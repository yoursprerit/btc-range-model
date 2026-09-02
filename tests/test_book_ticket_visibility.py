"""Unit tests for the as-published trade log's COVERAGE helpers
(app/overall_core.py) — ``pending_book_tickets`` and ``book_change_status``.

Synthetic books only (no network, no Streamlit).  Between them they answer
the question the 🧾 Daily trade log could not answer before: when the newest
row is days old, is the record stale or was there simply nothing to trade?

* a publish is only mapped onto a day STRICTLY AFTER its ``as_of``, so the
  book committed this morning is in no weight row yet — its ticket must
  still be listable (``pending``), and must not be listed twice once a bar
  covers it;
* a publish that repeats the previous book produces no ticket at all, so a
  run of them freezes the log while the archive keeps advancing — that run
  has to be countable and datable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _book(as_of, weights, cash=0.0):
    return {"schema": "overall-target-book/v1", "as_of": as_of,
            "profile": "Balanced", "book_mode": "live", "weights": weights,
            "cash_weight": cash, "exec_price": {}}


# Mon 2026-07-06 … Fri 2026-07-10 — five business days
IDX = pd.date_range("2026-07-06", periods=5, freq="D")

BOOKS = [_book("2026-07-06", {"AAA": 0.5}, 0.5),
         _book("2026-07-07", {"AAA": 0.5, "BBB": 0.3}, 0.2),   # BBB opened
         _book("2026-07-08", {"AAA": 0.5, "BBB": 0.3}, 0.2),   # unchanged
         _book("2026-07-09", {"BBB": 0.3}, 0.7)]               # AAA closed


# ── pending tickets — the newest publish, before its first bar ──────────────
def test_newest_publish_is_listed_before_its_first_bar_prints():
    # bars cover through 07-08, so the 07-09 book is in no weight row yet
    pend = oc.pending_book_tickets(BOOKS, "2026-07-08")
    assert [p["date"] for p in pend] == [pd.Timestamp("2026-07-09")]
    a = pend[0]["actions"][0]
    assert (a["key"], a["action"], a["signal_change"]) == ("AAA", "sell", True)
    assert pend[0]["pending"] is True
    # no P&L: the bar it will earn has not happened
    assert a["pnl"] is None and a["sold"] is None and a["entry_date"] is None
    assert np.isclose(pend[0]["cash0"], 0.2)
    assert np.isclose(pend[0]["cash1"], 0.7)
    assert np.isclose(pend[0]["gross"], 0.5)
    assert np.isclose(pend[0]["turnover"], 0.5 * (0.5 + 0.5))


def test_a_covered_publish_is_never_listed_pending_as_well():
    # the same ticket, once a bar AFTER 07-09 exists, belongs to the log
    assert oc.pending_book_tickets(BOOKS, "2026-07-10") == []
    # and the settled log does list it, dated at that publish
    rets = pd.DataFrame({"AAA": 0.0, "BBB": 0.0}, index=IDX)
    rep = oc.published_book_replay(rets, BOOKS, sata_daily=0.0)
    log = oc.daily_trade_log(rep["weights"], rep["sata"], IDX[0])
    assert pd.Timestamp("2026-07-09") in [d["date"] for d in log]


def test_pending_and_settled_partition_the_tickets_exactly():
    # bars → 07-08
    rets = pd.DataFrame({"AAA": 0.0, "BBB": 0.0}, index=IDX[:3])
    rep = oc.published_book_replay(rets, BOOKS, sata_daily=0.0)
    settled = {d["date"] for d in
               oc.daily_trade_log(rep["weights"], rep["sata"], IDX[0])}
    pending = {p["date"] for p in
               oc.pending_book_tickets(BOOKS, rep["weights"].index[-1])}
    assert settled == {pd.Timestamp("2026-07-07")}       # BBB opened
    assert pending == {pd.Timestamp("2026-07-09")}       # AAA closed
    assert not settled & pending


def test_quiet_publishes_yield_no_pending_row():
    # bars cover through 07-08, leaving the 07-08 book pending — but it
    # repeats 07-07, so there is nothing to list
    assert oc.pending_book_tickets(BOOKS[:3], "2026-07-08") == []


def test_live_sata_leg_is_cash_not_a_sleeve():
    # the live publisher parks idle capital as a real SATA weights entry
    books = [_book("2026-07-06", {"AAA": 0.5, "SATA": 0.5}),
             _book("2026-07-07", {"AAA": 0.3, "SATA": 0.7})]
    pend = oc.pending_book_tickets(books, "2026-07-06")
    assert [a["key"] for a in pend[0]["actions"]] == ["AAA"]     # no SATA row
    assert np.isclose(pend[0]["cash0"], 0.5)
    assert np.isclose(pend[0]["cash1"], 0.7)


def test_no_books_or_no_coverage():
    assert oc.pending_book_tickets([], "2026-07-08") == []
    assert oc.pending_book_tickets(BOOKS[:1], "2026-07-08") == []
    # nothing covered yet → every ticket after the first book is pending
    assert [p["date"] for p in oc.pending_book_tickets(BOOKS, None)] == \
        [pd.Timestamp("2026-07-09"), pd.Timestamp("2026-07-07")]


# ── change status — why the log's newest row can be days old ────────────────
def test_status_counts_the_unchanged_publishes_since_the_last_ticket():
    books = BOOKS + [_book("2026-07-10", {"BBB": 0.3}, 0.7),
                     _book("2026-07-11", {"BBB": 0.3}, 0.7)]
    st = oc.book_change_status(books)
    assert st["n_books"] == 6
    assert st["latest"] == pd.Timestamp("2026-07-11")
    assert st["last_change"] == pd.Timestamp("2026-07-09")
    assert st["quiet"] == 2
    assert st["quiet_from"] == pd.Timestamp("2026-07-10")


def test_status_resets_the_quiet_run_on_every_real_ticket():
    st = oc.book_change_status(BOOKS)
    assert (st["last_change"], st["quiet"]) == (pd.Timestamp("2026-07-09"), 0)
    assert st["quiet_from"] is None


def test_status_when_the_book_never_changed_and_when_there_are_no_books():
    same = [_book("2026-07-06", {"AAA": 1.0}),
            _book("2026-07-07", {"AAA": 1.0})]
    st = oc.book_change_status(same)
    assert st["last_change"] is None and st["quiet"] == 1
    assert oc.book_change_status([]) is None


def test_sub_min_delta_drift_is_not_a_ticket():
    drift = [_book("2026-07-06", {"AAA": 0.5000}, 0.5),
             _book("2026-07-07", {"AAA": 0.5002}, 0.4998)]
    assert oc.pending_book_tickets(drift, "2026-07-06") == []
    assert oc.book_change_status(drift)["quiet"] == 1


def test_parsed_replay_books_are_accepted_too():
    # published_book_replay's own book records (as_of Timestamp, SATA already
    # folded into `cash`) must feed the helpers unchanged — that is the shape
    # the app hands them
    rets = pd.DataFrame({"AAA": 0.0, "BBB": 0.0}, index=IDX[:3])
    rep = oc.published_book_replay(rets, BOOKS, sata_daily=0.0)
    assert [p["date"] for p in
            oc.pending_book_tickets(rep["books"], rep["weights"].index[-1])] \
        == [pd.Timestamp("2026-07-09")]
    assert oc.book_change_status(rep["books"])["last_change"] == \
        pd.Timestamp("2026-07-09")
