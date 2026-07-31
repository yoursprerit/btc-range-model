"""Unit tests for the as-published book replay (app/overall_core.py) — the
P&L section's "actual historical record" source.

Synthetic books + returns only (no network, no Streamlit): the
decided-at-previous-close convention, cost-basis anchor bar, business-day-only
SATA accrual on the cash remainder, daily-trim turnover accounting, unknown-key
handling, and the exact-additivity contract with ``pnl_attribution_replay``
(the P&L-by-asset toggle must sum to the headline curve in both sources).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


def _book(as_of, weights, cash, profile="Balanced", mode="live"):
    return {"schema": "overall-target-book/v1", "as_of": as_of,
            "profile": profile, "book_mode": mode, "weights": weights,
            "cash_weight": cash, "exec_price": {}}


def _rets(index, **cols):
    return pd.DataFrame(cols, index=index)


# Mon 2026-07-06 … Fri 2026-07-10 — five business days
IDX = pd.date_range("2026-07-06", periods=5, freq="D")


def test_anchor_bar_is_cost_basis_and_next_day_uses_first_book():
    rets = _rets(IDX, AAA=pd.Series(0.10, index=IDX), BBB=pd.Series(-0.02, index=IDX))
    books = [_book("2026-07-06", {"AAA": 0.5}, 0.5)]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    assert rep["ret"].iloc[0] == 0.0                       # anchor earns nothing
    # day 2 onwards: latest book strictly before the day → 0.5 × 10%
    assert np.allclose(rep["ret"].iloc[1:], 0.05)
    # the anchor row still DISPLAYS the book put on at that close
    assert rep["weights"].iloc[0]["AAA"] == 0.5
    assert rep["equity"].iloc[-1] == (1.05) ** 4


def test_book_switch_applies_from_next_bar():
    rets = _rets(IDX, AAA=pd.Series(0.01, index=IDX), BBB=pd.Series(0.04, index=IDX))
    books = [_book("2026-07-06", {"AAA": 1.0}, 0.0),
             _book("2026-07-08", {"BBB": 1.0}, 0.0)]       # trimmed AAA → BBB
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    r = rep["ret"]
    assert np.isclose(r.loc["2026-07-07"], 0.01)           # book(07-06)
    assert np.isclose(r.loc["2026-07-08"], 0.01)           # still book(07-06)
    assert np.isclose(r.loc["2026-07-09"], 0.04)           # book(07-08) from next bar
    assert np.isclose(r.loc["2026-07-10"], 0.04)


def test_cash_earns_sata_on_business_days_only():
    idx = pd.date_range("2026-07-09", periods=4, freq="D")   # Thu Fri Sat Sun
    rets = _rets(idx, AAA=pd.Series(0.0, index=idx))
    books = [_book("2026-07-09", {"AAA": 0.2}, 0.8)]
    rep = oc.published_book_replay(rets, books, sata_daily=0.001)
    r = rep["ret"]
    assert np.isclose(r.loc["2026-07-10"], 0.8 * 0.001)      # Friday accrues
    assert r.loc["2026-07-11"] == 0.0                        # weekend does not
    assert r.loc["2026-07-12"] == 0.0


def test_turnover_measures_the_daily_trims():
    books = [_book("2026-07-06", {"AAA": 0.6, "BBB": 0.4}, 0.0),
             _book("2026-07-07", {"AAA": 0.4, "BBB": 0.4}, 0.2)]   # trim AAA by 20pp
    rets = _rets(IDX, AAA=pd.Series(0.0, index=IDX), BBB=pd.Series(0.0, index=IDX))
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    b0, b1 = rep["books"]
    assert b0["turnover"] is None                            # nothing before it
    # ½ × (|Δ AAA| + |Δ BBB| + |Δ cash|) = ½ × (0.2 + 0 + 0.2)
    assert np.isclose(b1["turnover"], 0.2)


def test_unknown_book_key_is_reported_not_borrowed():
    rets = _rets(IDX, AAA=pd.Series(0.10, index=IDX))
    books = [_book("2026-07-06", {"AAA": 0.5, "GONE": 0.5}, 0.0)]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    assert rep["dropped"] == ["GONE"]
    assert np.allclose(rep["ret"].iloc[1:], 0.05)            # GONE earns nothing


def test_returns_none_when_nothing_replayable():
    rets = _rets(IDX, AAA=pd.Series(0.0, index=IDX))
    assert oc.published_book_replay(rets, [], sata_daily=0.0) is None
    late = [_book("2026-07-10", {"AAA": 1.0}, 0.0)]          # only the last bar
    assert oc.published_book_replay(rets, late, sata_daily=0.0) is None


def test_attribution_sums_exactly_to_the_curve():
    # the P&L-by-asset toggle reuses pnl_attribution_replay on the published
    # weights — per-key dollars + SATA must equal the curve's total return
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-07-01", periods=15, freq="D")
    rets = _rets(idx,
                 AAA=pd.Series(rng.normal(0, 0.02, 15), index=idx),
                 BBB=pd.Series(rng.normal(0, 0.03, 15), index=idx))
    books = [_book("2026-07-01", {"AAA": 0.5, "BBB": 0.3}, 0.2),
             _book("2026-07-04", {"AAA": 0.3, "BBB": 0.5}, 0.2),
             _book("2026-07-09", {"AAA": 0.7}, 0.3)]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0005)
    att = oc.pnl_attribution_replay(rets, rep["weights"], rep["sata"],
                                    idx[0], sata_daily=0.0005)
    total = rep["equity"].iloc[-1] / rep["equity"].iloc[0] - 1
    assert np.isclose(att["total"], total, atol=1e-12)
    assert np.isclose(sum(att["per_key"].values()) + att["sata"], total, atol=1e-9)


def test_live_book_sata_weight_is_the_cash_leg_not_a_sleeve():
    # live books park idle capital as a real SATA *weights* entry with
    # cash_weight 0 — it must accrue the coupon, not drop as an unknown key
    rets = _rets(IDX, AAA=pd.Series(0.0, index=IDX))
    books = [_book("2026-07-06", {"AAA": 0.4, "SATA": 0.6}, 0.0)]
    rep = oc.published_book_replay(rets, books, sata_daily=0.001)
    assert rep["dropped"] == []
    assert np.allclose(rep["ret"].iloc[1:], 0.6 * 0.001)     # all biz days
    assert rep["sata"].iloc[0] == 0.6
    assert rep["books"][0]["cash"] == 0.6


def test_version_labels_inline_wins_then_sidecar_then_unstamped():
    rets = _rets(IDX, AAA=pd.Series(0.0, index=IDX))
    b0 = _book("2026-07-06", {"AAA": 1.0}, 0.0)              # sidecar-labelled
    b1 = _book("2026-07-07", {"AAA": 1.0}, 0.0)              # unstamped
    b2 = dict(_book("2026-07-08", {"AAA": 1.0}, 0.0),
              strategy_version="v1", code_sha="abc123def4567890")  # inline
    vmap = {"2026-07-06": {"strategy_version": "pre-v1", "code_sha": "deadbeef"}}
    rep = oc.published_book_replay(rets, [b0, b1, b2], sata_daily=0.0,
                                   version_map=vmap)
    vs = [b["version"] for b in rep["books"]]
    assert vs == ["pre-v1", "unstamped", "v1"]
    assert rep["books"][0]["code_sha"] == "deadbeef"
    assert rep["books"][2]["code_sha"] == "abc123def456"     # trimmed to 12
    assert [(s["version"], s["n_books"]) for s in rep["version_spans"]] == \
        [("pre-v1", 1), ("unstamped", 1), ("v1", 1)]


def test_version_spans_group_consecutive_books():
    rets = _rets(IDX, AAA=pd.Series(0.0, index=IDX))
    books = [dict(_book(f"2026-07-0{d}", {"AAA": 1.0}, 0.0),
                  strategy_version=v)
             for d, v in [(6, "v1"), (7, "v1"), (8, "v2")]]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0)
    spans = rep["version_spans"]
    assert [(s["version"], s["n_books"]) for s in spans] == [("v1", 2), ("v2", 1)]
    assert spans[0]["start"] == pd.Timestamp("2026-07-06")
    assert spans[0]["end"] == pd.Timestamp("2026-07-07")
    assert spans[1]["start"] == pd.Timestamp("2026-07-08")


def test_only_version_filters_to_that_generation():
    rets = _rets(IDX, AAA=pd.Series(0.01, index=IDX))
    books = [dict(_book("2026-07-06", {"AAA": 1.0}, 0.0), strategy_version="v1"),
             dict(_book("2026-07-08", {"AAA": 0.5}, 0.5), strategy_version="v2")]
    rep = oc.published_book_replay(rets, books, sata_daily=0.0,
                                   only_version="v2")
    assert [b["version"] for b in rep["books"]] == ["v2"]
    # coverage re-anchors at the v2 book — no v1 days leak in
    assert rep["ret"].index[0] == pd.Timestamp("2026-07-08")
    assert np.allclose(rep["ret"].iloc[1:], 0.5 * 0.01)
    # filtering to a version with nothing replayable → None, not garbage
    assert oc.published_book_replay(rets, books, sata_daily=0.0,
                                    only_version="v9") is None


def test_strategy_version_single_source():
    # the badge module is the single source of truth; overall_core (and via it
    # the publisher's stamp) must re-export the identical value
    import strategy_version as sv
    assert oc.STRATEGY_VERSION == sv.STRATEGY_VERSION
    assert sv.STRATEGY_VERSION in sv.badge_caption()


def test_load_book_version_map_missing_file_is_empty(tmp_path):
    assert oc.load_book_version_map(tmp_path / "nope.json") == {}
    (tmp_path / "v.json").write_text(json.dumps(
        {"schema": "book-versions/v1",
         "books": {"2026-07-06": {"strategy_version": "pre-v1",
                                  "code_sha": "abc"}}}))
    m = oc.load_book_version_map(tmp_path / "v.json")
    assert m["2026-07-06"]["strategy_version"] == "pre-v1"


def test_load_published_books_reads_sorted_and_skips_junk(tmp_path):
    (tmp_path / "2026-07-08.json").write_text(
        json.dumps(_book("2026-07-08", {"AAA": 1.0}, 0.0)))
    (tmp_path / "2026-07-06.json").write_text(
        json.dumps(_book("2026-07-06", {"AAA": 0.5}, 0.5)))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "noweights.json").write_text(json.dumps({"as_of": "2026-07-07"}))
    books = oc.load_published_books(tmp_path)
    assert [b["as_of"] for b in books] == ["2026-07-06", "2026-07-08"]
