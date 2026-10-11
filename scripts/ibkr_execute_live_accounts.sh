#!/usr/bin/env bash
# Daily LIVE multi-account MOC executor (Option C) - the Linux twin of
# scripts/ibkr_execute_live_accounts.ps1.
#
# Pulls the latest published live book, then runs scripts/ibkr_execute_accounts.py
# --execute: every account in deploy/ibkr_accounts.json is traded together as
# MARKET-ON-CLOSE orders only (no fallback; past the 15:50 ET cutoff the session
# is skipped). Schedule ~2:15 PM US Central (3:15 PM ET).
#
#     CRON_TZ=America/Chicago
#     15 14 * * 1-5  /path/to/repo/scripts/ibkr_execute_live_accounts.sh >> /path/to/repo/logs/ibkr_live_cron.log 2>&1
#
# Env: IBKR_PYTHON, IBKR_BRANCH (main), IBKR_ACCOUNTS_CONFIG, IBKR_NO_PULL=1,
#      IBKR_NO_PUSH_REPORT=1, IBKR_TRADING_DISABLED=1, IBKR_PREVIEW=1 (no orders),
#      IBKR_ALERT_WEBHOOK. OVERALL_BOOK_SECRET is required.
# Exit: 0 ok | 1 account failed / not fully filled | 2 gateway/config | 3 skipped
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${IBKR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
BRANCH="${IBKR_BRANCH:-main}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_DIR}/ibkr_live_executor.log"; }

cd "${REPO_ROOT}"
log "----------------------------------------------"
log "starting IBKR LIVE multi-account MOC executor (branch=${BRANCH})"

if [ "${IBKR_NO_PULL:-0}" != "1" ]; then
  git fetch --quiet origin "${BRANCH}" 2>&1 | while read -r l; do log "$l"; done
  git merge --ff-only "origin/${BRANCH}" 2>&1 | while read -r l; do log "$l"; done \
    || log "WARN: ff-merge failed - using the book already on disk"
fi

ARGS=()
[ -n "${IBKR_ACCOUNTS_CONFIG:-}" ] && ARGS+=(--config "${IBKR_ACCOUNTS_CONFIG}")
[ "${IBKR_PREVIEW:-0}" != "1" ] && ARGS+=(--execute)

"${PYTHON}" scripts/ibkr_execute_accounts.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | while IFS= read -r l; do log "$l"; done
RC=${PIPESTATUS[0]}

case "${RC}" in
  0) log "all accounts OK" ;;
  3) log "session skipped (past the MOC cutoff / not a trading day / kill switch) - nothing traded" ;;
  2) log "ERROR: gateway or config problem - NOTHING was traded" ;;
  *) log "WARN: runner exited ${RC} - an account failed or did not fully fill" ;;
esac
log "done (runner exit ${RC})"
exit "${RC}"
