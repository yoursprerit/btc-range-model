# IBKR Paper Trading — Daily Rebalancer

Connect the **Overall Trading** strategy signals to an Interactive Brokers
**paper** account and rebalance it once per trading day. The paper account is
driven to the same allocation the Overall Streamlit app shows as **"Recommended
now (live-adjusted)"** — no signal logic is re-implemented; the rebalancer calls
the same `overall_core` engine.

> **Paper only.** The rebalancer refuses to run against any account whose id does
> not start with `DU` (IBKR's paper prefix) unless you explicitly pass
> `--allow-nonpaper`. `--dry-run` (no orders) is the default.

---

## How it works

```
overall_core.run_universe()                       every strategy, live
      → optimize_weights(… Aggressive …)          app-default risk blend
      → fetch_spot() + apply_spot()               live prices overlaid
      → live_exit_keys(include_entries=True)      drop trend-broken names
      → signal_gated_allocation(force_exit=…)     today's target weights
```

- **Targets** are portfolio weights per signal key. The undeployed remainder is
  **held as cash** (no SATA leg in the live book).
- **BTC → IBIT.** IBKR has no spot-Bitcoin product, so the BTC signal sleeve is
  traded via the IBIT spot-Bitcoin ETF. Every other name is a US-listed
  ETF/equity traded under its own ticker. See `scripts/ibkr_symbols.py`.
- **Sizing** uses each traded instrument's own live quote (the BTC sleeve is
  sized on IBIT's price, not spot BTC).
- **Reconcile** against current IBKR positions with a **no-trade band**
  (default 1% of net-liq) to suppress churn, then place **market orders
  sells-first** so freed capital funds the buys.

The full universe: BTC(→IBIT), MSTR, MSTU, GLDM, GDX, UGL, NUGT, SOXX, SOXL,
GRID, XLE, OIH, ERX, REMX, WGMI, PBW, ARTY.

---

## One-time setup

### 1. Python environment (on the host that will run IB Gateway)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt        # the strategy engine
.venv/bin/pip install -r requirements-ibkr.txt   # + ib_async broker layer
```

### 2. IB Gateway (paper)

1. Download **IB Gateway** (lighter than TWS):
   <https://www.interactivebrokers.com/en/trading/ib-api.php>
2. Log into your **paper** account (or a dedicated paper username).
3. **Configure → Settings → API → Settings**:
   - ☑ Enable ActiveX and Socket Clients
   - **Socket port `4002`** (IB Gateway paper default)
   - Add `127.0.0.1` to **Trusted IPs** (avoids the manual accept popup)
   - ☐ Read-Only API must be **unchecked** (orders need write access)

### 3. Unattended auto-login (IBC) — for the automated daily run

IB Gateway forces a daily restart and needs login each time; **IBC** automates
that so cron can rely on a live session.

1. Install IBC: <https://github.com/IbcAlpha/IBC>
2. Copy `deploy/ibc/config.ini.example` to your IBC config (e.g.
   `~/ibc/config.ini`) and fill in credentials + `TradingMode=paper`.
   **Do not commit the filled-in file** — it holds your password.
3. Launch IB Gateway under IBC at boot (and/or on a schedule shortly before the
   rebalance window). Set `AutoRestartTime` to the small hours so a restart
   never lands mid-rebalance.

> **2FA tip:** unattended login is simplest with a paper username that has 2FA
> disabled. If you must use IBKR Mobile 2FA, follow the IBC docs for
> second-factor handling.

---

## Running it

**Preview (default — places no orders).** Works even without a gateway; with one
connected it also diffs against your live positions:

```bash
.venv/bin/python scripts/ibkr_rebalance.py
```

**Execute against paper:**

```bash
.venv/bin/python scripts/ibkr_rebalance.py --execute
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--execute` | off (dry-run) | actually transmit orders |
| `--profile` | `Aggressive` (app default) | `Balanced` / `Growth` / `Aggressive` |
| `--band` | `0.01` | no-trade band as a fraction of net-liq |
| `--fractional` | off | allow fractional shares (default: whole shares) |
| `--port` | `4002` | IB Gateway API port (paper) |
| `--fill-timeout` | `60` | seconds to wait for each order leg to fill |
| `--force` | off | ignore the weekend/holiday & stale-signal guards |
| `--allow-nonpaper` | off | **danger** — disable the paper-account guard |

### Recommended validation before trusting automation
1. `scripts/ibkr_rebalance.py` (dry-run) — eyeball the target book against the
   Overall app's "Recommended now (live-adjusted)" panel; they should match.
2. Connect the gateway and dry-run again — check the order plan (deltas vs your
   current paper positions) looks sane.
3. `--execute` once by hand during US market hours; confirm fills and that
   resulting positions match the targets.
4. Only then enable the cron job.

---

## Automating the daily run

The market timing: signals are computed on completed daily bars and act on the
next bar, so rebalance shortly after the US open on trading days.

`scripts/ibkr_daily.sh` wraps a single `--execute` run with logging. Point cron
at it (the script and the Python guards both skip weekends/holidays):

```cron
CRON_TZ=America/New_York
45 9 * * 1-5  /path/to/repo/scripts/ibkr_daily.sh >> /path/to/repo/logs/ibkr_cron.log 2>&1
```

Override behaviour via env vars (see the script header): `IBKR_PYTHON`,
`IBKR_PROFILE`, `IBKR_BAND`, `IBKR_PORT`, `IBKR_EXTRA`.

Ensure IB Gateway (under IBC) is up **before** 09:45 ET and that its
`AutoRestartTime` sits well outside the rebalance window.

---

## Safety & limitations

- **Paper-account guard** blocks any non-`DU` account by default.
- **Dry-run is the default**; orders require `--execute`.
- **Freshness guard** aborts if the latest signal bar is > 4 days old (dead
  feed protection).
- Leveraged sleeves (MSTU 2×, SOXL 3×, UGL/NUGT/ERX 2×) are real ETFs and trade
  normally, but they are the volatile part of the book — paper-test thoroughly.
- Market orders fill at the prevailing price; for illiquid names consider adding
  a limit-order variant before ever considering live capital.
- This tool is for **paper** validation of the strategy's live behaviour. Going
  to real money is a separate, deliberate decision beyond this repo's scope.
