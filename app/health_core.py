"""🩺 Strategy Health — the decay-monitor compute layer (no Streamlit here).

Four monitors, computed nightly by ``scripts/build_strategy_health.py`` (a
step of the publish-target-book GitHub Action) and rendered by the read-only
``app/health_app.py`` page.  Design + thresholds: ``STRATEGY_HEALTH.md``.

  M1  Live-vs-replay tracking   — the daily return the PUBLISHED books
      (``data/overall/book_archive/``) actually earned close-to-close vs the
      walk-forward gated replay's theoretical return for the same days.
  M2  Drawdown tripwires        — per sleeve and per risk profile: current
      drawdown depth / underwater duration vs the reference (pre-live)
      history's maximum.  A breach falsifies the back-test's risk model.
  M3  Edge vs Buy & Hold        — per sleeve: rolling 126-bar strategy-minus-
      B&H return.  A sleeve's whole claim is *timing* alpha; persistent
      negative edge means the signal is dead even when absolute P&L is fine.
  M4  Trade expectancy          — per sleeve: mean return of the last 20
      closed trades, placed as a percentile inside a bootstrap distribution
      of same-size samples from the sleeve's earlier trades.

Every monitor emits one of four states — ``green`` / ``yellow`` / ``red`` /
``warming`` (⬜ = not enough data to judge; deliberately NOT green).  Breach
persistence (``first_breach_date``) is carried across snapshots here, in the
committed artifact — not in the UI — so every alarm is auditable in git
history and a one-day flicker can never page anyone.

Alarms feed the HUMAN review protocol (re-run the sleeve's honest-fill
re-sweep and decide); the adaptivity study (`OVERALL_ADAPTIVE_EVAL.md` V4)
showed automatic de-rating of cold sleeves is neutral-to-negative, so no
monitor here changes any weight.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "strategy-health/v1"

# ── thresholds — the ex-ante rules, versioned with the code ────────────────
# Documented in STRATEGY_HEALTH.md; change them there and here together.
THRESHOLDS = dict(
    # M2 drawdown tripwires
    dd_approach=0.80,        # |current DD| ≥ 80% of the reference max → yellow
    dd_min_ref_bars=250,     # reference history needed before the tripwire arms
    # M3 rolling edge vs Buy & Hold
    edge_window=126,         # ~6 months of daily bars
    edge_deadband_pp=0.5,    # |edge| inside this many pp is noise, not a breach
    edge_red_days=63,        # negative edge persisting ≥ this many calendar days → red
    # M4 trade expectancy
    exp_window=20,           # rolling trade window
    exp_min_ref=20,          # reference trades needed beyond the window
    exp_yellow_pctl=10.0,    # below this bootstrap percentile → yellow
    exp_red_pctl=5.0,        # below this AND persisted exp_red_days → red
    exp_red_days=14,
    boot_n=4000, boot_seed=7,
    # M1 tracking (published book vs replay), trailing window
    track_window=63,         # ~3 months
    track_min_days=21,       # live days needed before the gap monitor arms (⬜ until then)
    track_yellow=-0.015,     # trailing-window cum gap (live − replay) below → yellow
    track_red=-0.040,        # below → red
    # artifact size
    series_points=378,       # ~18 months of daily points kept per drill-down series
)

STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴", "warming": "⬜"}
_RANK = {"green": 0, "warming": 1, "yellow": 2, "red": 3}

# The first target book ever published (start of the live record); used as the
# reference/live split when the book archive is missing or empty.
LIVE_ANCHOR_FALLBACK = "2026-07-16"


# ════════════════════════════════════════════════════════════════════════
# Pure helpers (unit-tested in tests/test_health_core.py)
# ════════════════════════════════════════════════════════════════════════
def equity_curve(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret.fillna(0.0)).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float)
    return eq / eq.cummax() - 1.0


def underwater_stats(dd: pd.Series) -> dict:
    """Longest and current underwater spell, in CALENDAR days.

    A spell runs from the last at-high bar to each subsequent below-high bar;
    ``max_days`` is the longest such stretch anywhere in ``dd``; ``cur_days``
    is the ongoing one (0 when the curve sits at a high)."""
    if dd is None or not len(dd):
        return dict(max_days=0, cur_days=0)
    max_days = 0
    last_peak = dd.index[0]
    for date, v in dd.items():
        if v >= -1e-12:
            last_peak = date
        else:
            max_days = max(max_days, (date - last_peak).days)
    cur_days = (dd.index[-1] - last_peak).days if dd.iloc[-1] < -1e-12 else 0
    return dict(max_days=int(max_days), cur_days=int(cur_days))


def rolling_edge(strat_ret: pd.Series, bh_ret: pd.Series,
                 window: int = THRESHOLDS["edge_window"]) -> pd.Series:
    """Rolling ``window``-bar strategy-minus-B&H cumulative return, in
    percentage points.  Compounded via log-returns so the two legs subtract on
    the same basis."""
    df = pd.DataFrame({"s": strat_ret, "b": bh_ret}).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    ls = np.log1p(df["s"]).rolling(window).sum()
    lb = np.log1p(df["b"]).rolling(window).sum()
    return ((np.exp(ls) - np.exp(lb)) * 100.0).dropna()


def bootstrap_pctl(observed: float, pool, size: int,
                   n: int = THRESHOLDS["boot_n"],
                   seed: int = THRESHOLDS["boot_seed"]) -> float | None:
    """Percentile of ``observed`` inside the distribution of means of
    ``size``-draws (with replacement) from ``pool``.  Deterministic (fixed
    seed) so re-running a snapshot reproduces it.  None when the pool is
    smaller than ``size`` — the caller renders that as ⬜ warming up."""
    pool = np.asarray(list(pool), dtype=float)
    if len(pool) < size or not np.isfinite(observed):
        return None
    rng = np.random.default_rng(seed)
    means = pool[rng.integers(0, len(pool), size=(n, size))].mean(axis=1)
    return float((np.mean(means < observed) + 0.5 * np.mean(means == observed)) * 100.0)


def realized_book_returns(books: list[dict], rets: pd.DataFrame,
                          sata_daily: float) -> pd.Series:
    """Close-to-close daily returns the PUBLISHED target books earned.

    Day *d* is driven by the latest archived book with ``as_of`` strictly
    before *d* — the same decided-at-previous-close convention as the replay,
    so the two series differ only by *what was actually published* (weights,
    timing, missed publishes), which is exactly what M1 measures.  Sleeve
    returns come from the same ``returns_matrix`` the replay uses; the cash
    remainder earns the SATA coupon on business days.  Gross of costs, like
    the replay — real fills layer in once the executed-book history is long
    enough to compare.
    """
    parsed = []
    for b in books or []:
        try:
            parsed.append((pd.Timestamp(b["as_of"]),
                           {k: float(v) for k, v in (b.get("weights") or {}).items()},
                           float(b.get("cash_weight", 0.0) or 0.0)))
        except Exception:
            continue
    parsed.sort(key=lambda t: t[0])
    if not parsed or rets is None or rets.empty:
        return pd.Series(dtype=float)
    dates = np.array([t[0].to_datetime64() for t in parsed])
    idx = rets.index[rets.index > parsed[0][0]]
    out = {}
    R = rets.fillna(0.0)
    for d in idx:
        i = int(np.searchsorted(dates, pd.Timestamp(d).to_datetime64(), side="left")) - 1
        if i < 0:
            continue
        _, w, cash = parsed[i]
        r = sum(wt * float(R.at[d, k]) for k, wt in w.items() if k in R.columns)
        if d.dayofweek < 5:
            r += cash * sata_daily
        out[d] = r
    return pd.Series(out, dtype=float).sort_index()


def merge_first_breach(prev_breaches: dict | None, key: str,
                       breached: bool, today: str) -> str | None:
    """Carry a breach's first date across snapshots; clear it on recovery."""
    if not breached:
        return None
    return (prev_breaches or {}).get(key) or today


def worst_status(*statuses: str) -> str:
    """Aggregate: red > yellow > warming > green (warming never masks a flag)."""
    return max(statuses, key=lambda s: _RANK.get(s, 0)) if statuses else "green"


def _days_since(date_str: str | None, today: str) -> int:
    if not date_str:
        return 0
    return int((pd.Timestamp(today) - pd.Timestamp(date_str)).days)


def _round_series(s: pd.Series, nd: int, n_points: int) -> dict:
    tail = s.dropna().tail(n_points)
    return dict(dates=[str(d.date()) for d in tail.index],
                values=[round(float(v), nd) for v in tail.values])


def prev_breach_map(prev_snapshot: dict | None) -> dict:
    """Flatten a previous snapshot's ``first_breach_date`` fields into
    ``{"<scope>:<monitor>": date}`` for the persistence merge."""
    out = {}
    if not prev_snapshot:
        return out
    for row in prev_snapshot.get("sleeves") or []:
        for mon in ("dd", "edge", "expectancy"):
            d = (row.get(mon) or {}).get("first_breach_date")
            if d:
                out[f"{row['key']}:{mon}"] = d
    for name, p in (prev_snapshot.get("portfolio", {}).get("profiles") or {}).items():
        d = (p or {}).get("first_breach_date")
        if d:
            out[f"profile:{name}:dd"] = d
    d = ((prev_snapshot.get("portfolio", {}).get("tracking") or {})
         .get("first_breach_date"))
    if d:
        out["tracking:gap"] = d
    return out


# ════════════════════════════════════════════════════════════════════════
# Per-monitor evaluation
# ════════════════════════════════════════════════════════════════════════
def dd_health(daily_ret: pd.Series, anchor: pd.Timestamp,
              prev: dict, scope_key: str, today: str,
              th: dict = THRESHOLDS) -> dict:
    """M2 — current drawdown depth/duration vs the pre-``anchor`` reference."""
    eq = equity_curve(daily_ret)
    dd = drawdown_series(eq)
    ref_eq = eq.loc[:anchor]
    if len(ref_eq) < th["dd_min_ref_bars"]:
        return dict(status="warming", value=None, ref_mdd=None,
                    cur_uw_days=None, ref_max_uw_days=None,
                    first_breach_date=None,
                    need=f"{th['dd_min_ref_bars'] - len(ref_eq)} more reference bars")
    ref_dd = drawdown_series(ref_eq)
    ref_mdd = float(ref_dd.min())
    ref_uw = underwater_stats(ref_dd)["max_days"]
    cur = float(dd.iloc[-1])
    cur_uw = underwater_stats(dd)["cur_days"]
    breach = (cur < ref_mdd - 1e-12) or (ref_uw > 0 and cur_uw > ref_uw)
    approach = (cur < ref_mdd * th["dd_approach"]) or \
               (ref_uw > 0 and cur_uw > ref_uw * th["dd_approach"])
    status = "red" if breach else ("yellow" if approach else "green")
    fb = merge_first_breach(prev, f"{scope_key}:dd", breach or approach, today)
    return dict(status=status, value=round(cur, 4), ref_mdd=round(ref_mdd, 4),
                cur_uw_days=cur_uw, ref_max_uw_days=ref_uw,
                first_breach_date=fb, need=None)


def edge_health(strat_ret: pd.Series, bh_ret: pd.Series,
                prev: dict, scope_key: str, today: str,
                th: dict = THRESHOLDS) -> tuple[dict, pd.Series]:
    """M3 — rolling strategy-minus-B&H edge; persistent negative → red."""
    series = rolling_edge(strat_ret, bh_ret, th["edge_window"])
    if series.empty:
        return dict(status="warming", value_pp=None, first_breach_date=None,
                    need=f"{th['edge_window']} bars of joint history"), series
    cur = float(series.iloc[-1])
    breach = cur < -th["edge_deadband_pp"]     # hairline negatives are noise
    fb = merge_first_breach(prev, f"{scope_key}:edge", breach, today)
    status = "green" if not breach else \
        ("red" if _days_since(fb, today) >= th["edge_red_days"] else "yellow")
    return dict(status=status, value_pp=round(cur, 2),
                first_breach_date=fb, need=None), series


def expectancy_health(trade_rets: list[float], prev: dict, scope_key: str,
                      today: str, th: dict = THRESHOLDS) -> dict:
    """M4 — last-``exp_window`` trades' mean return as a bootstrap percentile
    of the sleeve's own earlier trades."""
    trades = [float(t) for t in (trade_rets or []) if np.isfinite(t)]
    w = th["exp_window"]
    need = w + th["exp_min_ref"] - len(trades)
    if need > 0:
        return dict(status="warming", pctl=None, window_mean=None,
                    n_trades=len(trades), first_breach_date=None,
                    need=f"{need} more closed trades")
    recent, pool = trades[-w:], trades[:-w]
    obs = float(np.mean(recent))
    pctl = bootstrap_pctl(obs, pool, w, th["boot_n"], th["boot_seed"])
    if pctl is None:
        return dict(status="warming", pctl=None, window_mean=round(obs, 4),
                    n_trades=len(trades), first_breach_date=None,
                    need="larger reference pool")
    breach = pctl < th["exp_yellow_pctl"]
    fb = merge_first_breach(prev, f"{scope_key}:expectancy", breach, today)
    status = "green" if not breach else \
        ("red" if pctl < th["exp_red_pctl"]
         and _days_since(fb, today) >= th["exp_red_days"] else "yellow")
    return dict(status=status, pctl=round(pctl, 1), window_mean=round(obs, 4),
                n_trades=len(trades), first_breach_date=fb, need=None)


def tracking_health(live: pd.Series, replay: pd.Series, prev: dict,
                    today: str, th: dict = THRESHOLDS) -> dict | None:
    """M1 — published-book realized returns vs the replay, cumulative and over
    the trailing ``track_window`` days."""
    df = pd.DataFrame({"live": live, "replay": replay}).dropna()
    if df.empty:
        return None
    cum_live = float(equity_curve(df["live"]).iloc[-1] - 1)
    cum_rep = float(equity_curve(df["replay"]).iloc[-1] - 1)
    gap_daily = df["live"] - df["replay"]
    gap_win = float(gap_daily.tail(th["track_window"]).sum())
    armed = len(df) >= th["track_min_days"]    # a days-old record is variance, not signal
    breach = armed and gap_win < th["track_yellow"]
    fb = merge_first_breach(prev, "tracking:gap", breach, today)
    status = "warming" if not armed else \
        ("red" if gap_win < th["track_red"] else ("yellow" if breach else "green"))
    n_pts = th["series_points"]
    return dict(
        status=status, n_days=int(len(df)),
        first_date=str(df.index[0].date()), last_date=str(df.index[-1].date()),
        cum_live=round(cum_live, 4), cum_replay=round(cum_rep, 4),
        cum_gap=round(cum_live - cum_rep, 4),
        gap_window=round(gap_win, 4), window=th["track_window"],
        mean_daily_gap_bps=round(float(gap_daily.mean()) * 1e4, 2),
        first_breach_date=fb,
        series=dict(dates=[str(d.date()) for d in df.index][-n_pts:],
                    live=[round(v, 5) for v in
                          equity_curve(df["live"]).values][-n_pts:],
                    replay=[round(v, 5) for v in
                            equity_curve(df["replay"]).values][-n_pts:]))


# ════════════════════════════════════════════════════════════════════════
# Snapshot assembly
# ════════════════════════════════════════════════════════════════════════
def _sleeve_trade_rets(res: dict) -> list[float]:
    """Closed-trade returns, oldest→newest by exit date.  ``trade_log`` holds
    dicts; the BTC CT engine's ``trades`` are dicts too — a bare array of
    returns (undated) is accepted as already-ordered."""
    r = res.get("r") or {}
    log = r.get("trade_log")
    if not log:
        raw = r.get("trades")
        log = list(raw) if raw is not None else []
    dicts = [t for t in log if isinstance(t, dict)]
    if dicts:
        dicts.sort(key=lambda t: pd.Timestamp(t.get("exit_date", "1970-01-01")))
        return [float(t["ret"]) for t in dicts if t.get("ret") is not None]
    return [float(t) for t in log if not isinstance(t, dict)]


def sleeve_health(res: dict, anchor: pd.Timestamp, prev: dict, today: str,
                  th: dict = THRESHOLDS) -> dict:
    key = res["key"]
    strat_ret = res["ret"]
    bh_eq = pd.Series(np.asarray(res["r"]["bh"], float),
                      index=pd.DatetimeIndex(res["dates"]))
    bh_ret = bh_eq.pct_change().dropna()
    dd = dd_health(strat_ret, anchor, prev, key, today, th)
    edge, edge_series = edge_health(strat_ret, bh_ret, prev, key, today, th)
    exp = expectancy_health(_sleeve_trade_rets(res), prev, key, today, th)
    status = worst_status(dd["status"], edge["status"], exp["status"])
    dd_series = drawdown_series(equity_curve(strat_ret))
    n = th["series_points"]
    return dict(
        key=key, name=res.get("name", key), parent=res.get("parent", key),
        kind=res.get("kind", "core"), emoji=res.get("emoji", ""),
        as_of=str(pd.Timestamp(res["as_of"]).date()),
        n_trades=int(res.get("n_trades", 0)),
        dd=dd, edge=edge, expectancy=exp, status=status,
        series=dict(dates=_round_series(dd_series, 4, n)["dates"],
                    dd=_round_series(dd_series, 4, n)["values"],
                    edge_dates=_round_series(edge_series, 2, n)["dates"]
                    if not edge_series.empty else [],
                    edge=_round_series(edge_series, 2, n)["values"]
                    if not edge_series.empty else []))


def build_snapshot(results: list[dict], replays: dict[str, dict],
                   books: list[dict], prev_snapshot: dict | None,
                   sata_daily: float, rets: pd.DataFrame | None = None,
                   now_utc: str | None = None,
                   th: dict = THRESHOLDS) -> dict:
    """Assemble the full ``strategy_health.json`` payload.

    ``replays``: profile name → ``walkforward_gated_replay()`` output (at
    least the default profile; whatever is supplied is monitored).
    ``books``: parsed archive payloads (``load_books``).  ``prev_snapshot``
    supplies breach persistence."""
    if rets is None:
        rets = pd.DataFrame({r["key"]: r["ret"] for r in results}).sort_index()
    prev = prev_breach_map(prev_snapshot)
    anchor = pd.Timestamp(min((b.get("as_of") for b in books or []),
                              default=LIVE_ANCHOR_FALLBACK))
    now = pd.Timestamp(now_utc) if now_utc else pd.Timestamp.now("UTC")
    today = str(now.date())

    sleeves = sorted((sleeve_health(r, anchor, prev, today, th)
                      for r in results),
                     key=lambda s: _RANK.get(s["status"], 0), reverse=True)

    profiles = {}
    replay_default = None
    for name, rep in (replays or {}).items():
        if rep is None:
            continue
        if replay_default is None:
            replay_default = rep
        p = dd_health(rep["ret"], anchor, prev, f"profile:{name}", today, th)
        profiles[name] = dict(metrics={k: round(float(v), 4) for k, v in
                                       (rep.get("metrics") or {}).items()}, **p)

    tracking = None
    if replay_default is not None:
        live = realized_book_returns(books, rets, sata_daily)
        if len(live):
            tracking = tracking_health(live, replay_default["ret"], prev,
                                       today, th)

    n_red = sum(s["status"] == "red" for s in sleeves) + \
        sum(p["status"] == "red" for p in profiles.values()) + \
        (1 if tracking and tracking["status"] == "red" else 0)
    n_yellow = sum(s["status"] == "yellow" for s in sleeves) + \
        sum(p["status"] == "yellow" for p in profiles.values()) + \
        (1 if tracking and tracking["status"] == "yellow" else 0)
    n_warming = sum(s["status"] == "warming" for s in sleeves)
    status = "red" if n_red else ("yellow" if n_yellow else "green")
    headline = ("All monitored sleeves healthy" if status == "green" else
                f"{n_red} alarm{'s' if n_red != 1 else ''} · "
                f"{n_yellow} warning{'s' if n_yellow != 1 else ''}")

    as_of = max((s["as_of"] for s in sleeves), default=today)
    return dict(
        schema=SCHEMA,
        generated_at_utc=now.isoformat(timespec="seconds"),
        as_of=as_of, live_anchor=str(anchor.date()), thresholds=dict(th),
        verdict=dict(status=status, n_red=n_red, n_yellow=n_yellow,
                     n_warming=n_warming, n_sleeves=len(sleeves),
                     headline=headline),
        portfolio=dict(profiles=profiles, tracking=tracking),
        sleeves=sleeves)


# ════════════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════════════
def load_books(archive_dir: Path) -> list[dict]:
    books = []
    for p in sorted(Path(archive_dir).glob("*.json")):
        try:
            books.append(json.loads(p.read_text()))
        except Exception:
            continue
    return books


def load_prev(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def history_rows(snap: dict) -> pd.DataFrame:
    """One compact row per (date, scope) — the append-only decay time series."""
    d = snap["as_of"]
    rows = [dict(date=d, scope="sleeve", key=s["key"], status=s["status"],
                 dd=s["dd"]["value"], edge_pp=s["edge"]["value_pp"],
                 exp_pctl=s["expectancy"]["pctl"],
                 n_trades=s["expectancy"]["n_trades"])
            for s in snap["sleeves"]]
    for name, p in (snap["portfolio"]["profiles"] or {}).items():
        rows.append(dict(date=d, scope="profile", key=name, status=p["status"],
                         dd=p["value"], edge_pp=None, exp_pctl=None,
                         n_trades=None))
    t = snap["portfolio"]["tracking"]
    if t:
        rows.append(dict(date=d, scope="tracking", key="book_vs_replay",
                         status=t["status"], dd=None,
                         edge_pp=round(t["gap_window"] * 100, 2),
                         exp_pctl=None, n_trades=t["n_days"]))
    return pd.DataFrame(rows)


def write_artifacts(snap: dict, out_json: Path, hist_csv: Path) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(snap, indent=1, default=str))
    new = history_rows(snap)
    hist_csv = Path(hist_csv)
    if hist_csv.exists():
        old = pd.read_csv(hist_csv, dtype={"date": str})
        old = old[old["date"] != snap["as_of"]]        # idempotent re-publish
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["date", "scope", "key"]).to_csv(hist_csv, index=False)
