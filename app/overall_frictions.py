"""Brokerage costs and capital-gains taxes for the walk-forward replay.

``simulate_frictions`` re-plays the replay's daily target-weight book in
DOLLARS, lot by lot, so it can charge what a real account would pay:

* **IBKR Pro Fixed** commissions on every order (US stocks / ETFs) plus the
  regulatory pass-through fees on sales;
* **federal capital-gains tax** with a running per-calendar-year tally of
  short-term and long-term realized gains/losses (FIFO lots, > 1 year = long
  term), loss netting, the $3,000 ordinary-income offset and carry-forwards.

Timing convention (matches ``pnl_daily_replay``): the anchor bar is the cost
basis; weights on bar *t* earn bar *t*'s return, so the book is rebalanced at
the close of *t-1*.  Idle capital sits in the SATA sleeve like the replay's.

Deliberate simplifications (surfaced in the UI caption): federal tax only (no
state), 2025 brackets applied to every year, wash-sale rules ignored, the SATA
coupon is treated as price appreciation, and unrealized gains at the end of the
window are not taxed.  Tax is accrued as a liability the moment gains are
realized (it reduces displayed portfolio value) and paid out of the SATA/cash
leg at each calendar year-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── IBKR Pro Fixed (US stocks/ETFs) ─────────────────────────────────────────
IBKR_PER_SHARE = 0.005          # $/share
IBKR_MIN_ORDER = 1.00           # $ minimum per order
IBKR_MAX_PCT = 0.01             # commission capped at 1% of trade value
SEC_FEE_RATE = 27.80e-6         # of sale value (sales only)
FINRA_TAF_PER_SHARE = 0.000166  # $/share sold …
FINRA_TAF_MAX = 8.30            # … capped per trade
DEFAULT_PX = 50.0               # share price when a sleeve has no price history
MIN_REBAL_FRAC = 0.0025         # ignore resizes smaller than 0.25% of equity

# ── Federal tax, married filing jointly, 2025 ───────────────────────────────
ORD_BRACKETS = [(0, .10), (23_850, .12), (96_950, .22), (206_700, .24),
                (394_600, .32), (501_050, .35), (751_600, .37)]
LTCG_0_TOP = 96_700
LTCG_15_TOP = 600_050
NIIT_RATE = 0.038               # MAGI > $250k MFJ — always true at $300-350k
LOSS_OFFSET = 3_000.0
INCOME_MIN, INCOME_MAX, INCOME_DEFAULT = 300_000, 350_000, 325_000


def order_cost(value: float, px: float, sell: bool) -> float:
    """Dollar cost of one order of ``value`` dollars at share price ``px``."""
    value = abs(value)
    if value <= 0:
        return 0.0
    shares = value / max(px, 1e-9)
    comm = min(max(IBKR_PER_SHARE * shares, IBKR_MIN_ORDER), IBKR_MAX_PCT * value)
    if sell:
        comm += SEC_FEE_RATE * value + min(FINRA_TAF_PER_SHARE * shares, FINRA_TAF_MAX)
    return comm


def _ord_tax(income: float) -> float:
    tax = 0.0
    for i, (lo, rate) in enumerate(ORD_BRACKETS):
        hi = ORD_BRACKETS[i + 1][0] if i + 1 < len(ORD_BRACKETS) else float("inf")
        if income > lo:
            tax += (min(income, hi) - lo) * rate
    return tax


def _lt_tax(lt_gain: float, stack_base: float) -> float:
    """LTCG tax on ``lt_gain`` stacked on top of ``stack_base`` of income."""
    tax, lo = 0.0, stack_base
    for top, rate in ((LTCG_0_TOP, 0.0), (LTCG_15_TOP, 0.15), (float("inf"), 0.20)):
        room = max(top - lo, 0.0)
        part = min(lt_gain, room)
        tax += part * rate
        lt_gain -= part
        lo = max(lo, top)
        if lt_gain <= 0:
            break
    return tax


def net_year(st: float, lt: float, carry_st: float, carry_lt: float) -> dict:
    """Net a year's realized ST/LT (``st``/``lt`` are signed sums) against the
    prior-year loss carry-forwards (positive numbers).  Returns the netted
    ST/LT gains and the net ST/LT losses (all >= 0, at most one side each)."""
    st -= carry_st
    lt -= carry_lt
    if st < 0 < lt:                      # ST loss shelters LT gain
        lt += st
        st = 0.0 if lt >= 0 else lt      # leftover loss keeps ST character
        lt = max(lt, 0.0)
    elif lt < 0 < st:                    # LT loss shelters ST gain
        st += lt
        lt = 0.0 if st >= 0 else st
        st = max(st, 0.0)
    return dict(st=max(st, 0.0), lt=max(lt, 0.0),
                st_loss=max(-st, 0.0), lt_loss=max(-lt, 0.0))


def year_tax(st: float, lt: float, carry_st: float, carry_lt: float,
             base_income: float) -> dict:
    """Federal tax owed on a year's realized gains stacked on ``base_income``
    of other ordinary income, plus the carry-forwards into next year."""
    n = net_year(st, lt, carry_st, carry_lt)
    gst, glt = n["st"], n["lt"]
    loss = n["st_loss"] + n["lt_loss"]
    ord_inc = _ord_tax(base_income + gst) - _ord_tax(base_income)
    lt_tax = _lt_tax(glt, base_income + gst)
    niit = NIIT_RATE * (gst + glt)
    benefit, cf_st, cf_lt = 0.0, n["st_loss"], n["lt_loss"]
    if loss > 0:
        use = min(loss, LOSS_OFFSET)
        benefit = _ord_tax(base_income) - _ord_tax(base_income - use)
        take = min(cf_st, use)               # ST losses are used first
        cf_st -= take
        cf_lt -= min(cf_lt, use - take)
    return dict(tax=ord_inc + lt_tax + niit - benefit, carry_st=cf_st,
                carry_lt=cf_lt, net_st=gst - n["st_loss"], net_lt=glt - n["lt_loss"])


def _sell_fifo(lots: list, amount: float, today: pd.Timestamp):
    """Sell ``amount`` dollars of value FIFO.  Returns (st_gain, lt_gain)."""
    st = lt = 0.0
    while amount > 1e-9 and lots:
        d, basis, val = lots[0]
        take = min(amount, val)
        frac = take / val if val > 0 else 1.0
        gain = take - basis * frac
        if (today - d).days > 365:
            lt += gain
        else:
            st += gain
        if frac >= 1.0 - 1e-12:
            lots.pop(0)
        else:
            lots[0] = [d, basis * (1 - frac), val - take]
        amount -= take
    return st, lt


def simulate_frictions(rets: pd.DataFrame, weights: pd.DataFrame, sata_w: pd.Series,
                       prices: pd.DataFrame | None, start, end=None, *,
                       portfolio_value: float = 100_000.0,
                       costs: bool = False, taxes: bool = False,
                       income: float = INCOME_DEFAULT,
                       sata_daily: float = 0.0) -> dict | None:
    """Dollar re-play of the target-weight book from ``start`` with the chosen
    frictions.  Returns ``net`` (value per $1 at the anchor, indexed from the
    anchor bar), ``gross`` (same book, no frictions), totals, and the
    per-calendar-year tax tally (``years``).  ``None`` with < 2 bars."""
    idx = rets.index
    a = idx.searchsorted(pd.Timestamp(start))
    b = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end), side="right")
    if b - a < 2:
        return None
    cols = list(rets.columns) + ["__SATA__"]
    R = np.nan_to_num(rets.to_numpy(float))
    W = np.column_stack([weights.reindex(idx).fillna(0.0).to_numpy(float),
                         sata_w.reindex(idx).fillna(1.0).to_numpy(float)])
    biz = np.asarray(idx.dayofweek < 5)
    Px = (prices.reindex(idx).ffill().bfill().to_numpy(float)
          if prices is not None else np.full((len(idx), len(cols) - 1), np.nan))
    Px = np.where(np.isfinite(Px) & (Px > 0), Px, DEFAULT_PX)

    lots = {j: [] for j in range(len(cols))}
    val = np.zeros(len(cols))
    equity = float(portfolio_value)
    st_y = lt_y = 0.0
    carry_st = carry_lt = 0.0
    accrued = 0.0                        # YTD liability not yet paid
    total_comm = total_tax = 0.0
    n_orders = 0
    years: list[dict] = []
    y_stats = dict(st_gain=0.0, st_loss=0.0, lt_gain=0.0, lt_loss=0.0, comm=0.0)

    def _px(j, t):
        return DEFAULT_PX if j == len(cols) - 1 else float(Px[t, j])

    def rebalance(t_dec: int, t_next: int, day: pd.Timestamp, first: bool):
        """Trade at close ``t_dec`` into the weights that earn bar ``t_next``."""
        nonlocal equity, st_y, lt_y, total_comm, n_orders
        tgt = W[t_next] * equity
        delta = tgt - val
        trade = np.zeros(len(cols), bool)
        for j in range(len(cols) - 1):
            small = abs(delta[j]) < MIN_REBAL_FRAC * equity
            if costs and not first and small and tgt[j] > 0 and val[j] > 0:
                continue
            if abs(delta[j]) > 1e-6:
                trade[j] = True
        comm = 0.0
        sells = [j for j in range(len(cols) - 1) if trade[j] and delta[j] < 0]
        buys = [j for j in range(len(cols) - 1) if trade[j] and delta[j] > 0]
        for j in sells:
            amt = -delta[j]
            if taxes:
                s, l = _sell_fifo(lots[j], amt, day)
                st_y += s
                lt_y += l
                y_stats["st_gain" if s >= 0 else "st_loss"] += abs(s)
                y_stats["lt_gain" if l >= 0 else "lt_loss"] += abs(l)
            else:
                _sell_fifo(lots[j], amt, day)
            val[j] -= amt
            if costs:
                comm += order_cost(amt, _px(j, t_dec), True)
                n_orders += 1
        # the SATA leg is the residual: it funds buys, absorbs sells and pays costs
        for j in buys:
            amt = delta[j]
            lots[j].append([day, amt, amt])
            val[j] += amt
            if costs:
                comm += order_cost(amt, _px(j, t_dec), False)
                n_orders += 1
        sj = len(cols) - 1
        want = equity - val[:sj].sum() - comm
        d_s = want - val[sj]
        if d_s > 1e-9:
            lots[sj].append([day, d_s, d_s])
        elif d_s < -1e-9:
            s, l = _sell_fifo(lots[sj], -d_s, day)
            if taxes:
                st_y += s
                lt_y += l
                y_stats["st_gain" if s >= 0 else "st_loss"] += abs(s)
                y_stats["lt_gain" if l >= 0 else "lt_loss"] += abs(l)
        val[sj] = want
        equity -= comm
        total_comm += comm
        y_stats["comm"] += comm

    def liability():
        if not taxes:
            return 0.0
        return year_tax(st_y, lt_y, carry_st, carry_lt, income)["tax"]

    dates, net_v, gross_v = [idx[a]], [1.0], [1.0]
    gross_eq = float(portfolio_value)
    rebalance(a, a + 1, idx[a], first=True)
    cur_year = idx[a].year
    for t in range(a + 1, b):
        day = idx[t]
        if day.year != cur_year:                       # year-end settlement
            if taxes:
                res = year_tax(st_y, lt_y, carry_st, carry_lt, income)
                due = res["tax"]
                years.append(dict(year=cur_year, tax=res["tax"], **y_stats,
                                  net_st=res["net_st"], net_lt=res["net_lt"],
                                  carry_in_st=carry_st, carry_in_lt=carry_lt))
                carry_st, carry_lt = res["carry_st"], res["carry_lt"]
                pay = min(due, max(equity, 0.0)) if due > 0 else due
                if pay != 0.0 and equity > 0:
                    sj = len(cols) - 1
                    from_sata = min(pay, val[sj]) if pay > 0 else pay
                    if from_sata > 0:
                        _sell_fifo(lots[sj], from_sata, idx[t - 1])
                    elif from_sata < 0:
                        lots[sj].append([idx[t - 1], -from_sata, -from_sata])
                    val[sj] -= from_sata
                    rest = pay - from_sata
                    if rest > 1e-9 and val.sum() > 0:  # pro-rata scale the rest
                        f = 1 - rest / val.sum()
                        for j in lots:
                            lots[j] = [[d, bs * f, v * f] for d, bs, v in lots[j]]
                        val *= f
                    equity -= pay
                total_tax += res["tax"]
                st_y = lt_y = 0.0
                y_stats = dict(st_gain=0.0, st_loss=0.0, lt_gain=0.0,
                               lt_loss=0.0, comm=0.0)
            cur_year = day.year
        # bar t's return on the book decided at t-1
        for j in range(len(cols) - 1):
            if val[j] > 0 and R[t, j] != 0.0:
                g = 1.0 + R[t, j]
                val[j] *= g
                for lot in lots[j]:
                    lot[2] *= g
        sj = len(cols) - 1
        if val[sj] > 0 and biz[t] and sata_daily:
            g = 1.0 + sata_daily
            val[sj] *= g
            for lot in lots[sj]:
                lot[2] *= g
        equity = float(val.sum())
        gross_eq *= 1.0 + float(np.dot(W[t, :-1], R[t])
                                + (W[t, -1] * sata_daily if biz[t] else 0.0))
        if t + 1 < len(idx) and t + 1 < b:
            rebalance(t, t + 1, day, first=False)
        accrued = max(liability(), 0.0)
        dates.append(day)
        net_v.append((equity - accrued) / portfolio_value)
        gross_v.append(gross_eq / portfolio_value)

    cur = year_tax(st_y, lt_y, carry_st, carry_lt, income) if taxes else None
    if taxes and (st_y or lt_y or y_stats["comm"] or not years
                  or years[-1]["year"] != cur_year):
        years.append(dict(year=cur_year, tax=cur["tax"], **y_stats,
                          net_st=cur["net_st"], net_lt=cur["net_lt"],
                          carry_in_st=carry_st, carry_in_lt=carry_lt,
                          partial=True))
        total_tax += cur["tax"]
    return dict(net=pd.Series(net_v, index=pd.DatetimeIndex(dates)),
                gross=pd.Series(gross_v, index=pd.DatetimeIndex(dates)),
                commissions=total_comm, taxes=total_tax, orders=n_orders,
                years=years)
