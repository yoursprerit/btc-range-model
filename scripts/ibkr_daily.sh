#!/usr/bin/env bash
# Daily IBKR paper rebalance — cron wrapper.
#
# Runs the rebalancer once, appending a timestamped log. Intended to be fired by
# cron on US trading days (the Python script itself also guards against weekends
# and holidays, so a stray weekend fire is a safe no-op).
#
# Example crontab (weekdays 14:30 America/Chicago = 2:30 PM US Central /
# 3:30 PM ET — 30 minutes before the equity close, so the strategy's "act on
# the next bar" executes late in that session). Set CRON_TZ so the schedule
# tracks US Central (and its DST switch) regardless of the host clock:
#
#     CRON_TZ=America/Chicago
#     30 14 * * 1-5  /path/to/repo/scripts/ibkr_daily.sh >> /path/to/repo/logs/ibkr_cron.log 2>&1
#
# Prerequisites (see IBKR_PAPER_TRADING.md):
#   * IB Gateway logged into the PAPER account (via IBC for unattended login)
#   * API enabled, socket port 4002, 127.0.0.1 a trusted IP
#   * a venv with requirements.txt + requirements-ibkr.txt installed
set -euo pipefail

# Resolve the repo root from this script's location (…/repo/scripts/ibkr_daily.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Point PYTHON at your venv's interpreter (override via the env var if needed).
PYTHON="${IBKR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

# Rebalancer flags — override from the environment without editing this file:
#   IBKR_PROFILE  risk profile (default: the app default — currently Balanced)
#   IBKR_BAND     no-trade band as a fraction of net-liq (default 0.01)
#   IBKR_PORT     IB Gateway API port (default 4002 = paper)
#   IBKR_EXTRA    any extra flags (e.g. "--fractional")
PROFILE="${IBKR_PROFILE:-}"
BAND="${IBKR_BAND:-0.01}"
PORT="${IBKR_PORT:-4002}"
EXTRA="${IBKR_EXTRA:-}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "──────────────────────────────────────────────────────────────"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting IBKR paper rebalance"

ARGS=(--execute --band "${BAND}" --port "${PORT}")
[ -n "${PROFILE}" ] && ARGS+=(--profile "${PROFILE}")
# shellcheck disable=SC2206
[ -n "${EXTRA}" ] && ARGS+=(${EXTRA})

cd "${REPO_ROOT}"
"${PYTHON}" scripts/ibkr_rebalance.py "${ARGS[@]}"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] done"
