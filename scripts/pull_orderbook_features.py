"""Pull free Binance order-book / derivatives microstructure history and build
12:00-UTC-anchored daily features (matches data/backtest BTC bars).

Sources (data.binance.vision, public, no key):
  futures/um bookDepth   -> +/-1..5% depth imbalance (2023-01+ only; NaN before)
  futures/um metrics     -> open interest, long/short ratios, taker L/S vol ratio
  futures/um fundingRate -> 8h funding
  futures/um premiumIndexKlines 1h -> perp premium
  spot klines 1h         -> taker-buy flow

Output: data/orderbook/hourly_ob.csv  (hourly, UTC, no look-ahead: row t = data in [t, t+1h))
"""
import io, sys, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

B = "https://data.binance.vision/data"
S = requests.Session()
START, END = dt.date(2021, 12, 1), dt.date.today() - dt.timedelta(days=1)


def get_zip(url, header="infer"):
    for _ in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]), header=header)
        except Exception:
            pass
    return None


def depth_day(d):
    df = get_zip(f"{B}/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{d}.zip")
    if df is None or df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    p = df.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    out = pd.DataFrame(index=p.index)
    for k in (1, 2, 5):
        bid, ask = p.get(-float(k)), p.get(float(k))
        if bid is None or ask is None:
            continue
        out[f"imb{k}"] = (bid - ask) / (bid + ask)
        out[f"depth{k}"] = bid + ask
    return out.resample("1h").mean()


def metrics_day(d):
    df = get_zip(f"{B}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip")
    if df is None or df.empty:
        return None
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df.set_index("create_time").drop(columns="symbol")
    return df.resample("1h").last()


def month_iter():
    m = dt.date(START.year, START.month, 1)
    while m <= END:
        yield m
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)


def klines_month(kind, m):
    tag = f"{m:%Y-%m}"
    if kind == "spot":
        url = f"{B}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{tag}.zip"
    elif kind == "perp":
        url = f"{B}/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{tag}.zip"
    else:
        url = f"{B}/futures/um/monthly/premiumIndexKlines/BTCUSDT/1h/BTCUSDT-1h-{tag}.zip"
    df = get_zip(url, header=None)
    if df is None:
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():     # header row present
        df = df.iloc[1:]
    df = df.iloc[:, :11].astype(float); df.columns = cols
    ot = df["open_time"]
    unit = "us" if ot.iloc[0] > 1e14 else "ms"
    df.index = pd.to_datetime(ot, unit=unit)
    return df


def funding_month(m):
    df = get_zip(f"{B}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{m:%Y-%m}.zip")
    if df is None:
        return None
    df.index = pd.to_datetime(df["calc_time"], unit="ms").dt.floor("1h")
    return df[["last_funding_rate"]]


def run(fn, items, w=10):
    with ThreadPoolExecutor(w) as ex:
        res = list(ex.map(fn, items))
    return [r for r in res if r is not None]


if __name__ == "__main__":
    days = [START + dt.timedelta(n) for n in range((END - START).days + 1)]
    months = list(month_iter())
    import os
    if os.path.exists("/tmp/_ob_cache2.pkl"):
        depth, met = pd.read_pickle("/tmp/_ob_cache2.pkl")
    else:
        print("depth..."); depth = pd.concat(run(depth_day, days)).sort_index()
        print("metrics..."); met = pd.concat(run(metrics_day, days)).sort_index()
        pd.to_pickle((depth, met), "/tmp/_ob_cache2.pkl")
    print("klines..."); 
    spot = pd.concat(run(lambda m: klines_month("spot", m), months)).sort_index()
    perp = pd.concat(run(lambda m: klines_month("perp", m), months)).sort_index()
    prem = pd.concat(run(lambda m: klines_month("prem", m), months)).sort_index()
    fund = pd.concat(run(funding_month, months)).sort_index()
    h = pd.DataFrame(index=pd.date_range(START, END + dt.timedelta(1), freq="1h", inclusive="left"))
    h = h.join(depth).join(met)
    h["spot_open"] = spot["open"]; h["spot_high"] = spot["high"]; h["spot_low"] = spot["low"]
    h["spot_close"] = spot["close"]; h["spot_vol"] = spot["volume"]
    h["spot_tbuy"] = spot["taker_buy_volume"]; h["spot_trades"] = spot["count"]
    h["perp_close"] = perp["close"]; h["perp_vol"] = perp["volume"]
    h["perp_tbuy"] = perp["taker_buy_volume"]
    h["prem"] = prem["close"]
    h["funding"] = fund["last_funding_rate"].reindex(h.index).ffill()
    h.index.name = "ts_utc"
    h.to_csv("data/orderbook/hourly_ob.csv")
    print(h.shape, h.index.min(), h.index.max()); print(h.notna().mean().round(3))
