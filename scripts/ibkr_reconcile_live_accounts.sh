#!/usr/bin/env bash
# After-the-close reconcile for the live multi-account MOC run - the Linux twin of
# scripts/ibkr_reconcile_live_accounts.ps1. Schedule ~4:15 PM US Eastern:
#
#     CRON_TZ=America/New_York
#     15 16 * * *  /path/to/repo/scripts/ibkr_reconcile_live_accounts.sh >> /path/to/repo/logs/ibkr_reconcile_cron.log 2>&1
#
# Env: IBKR_PYTHON, IBKR_ACCOUNTS_CONFIG, IBKR_RECONCILE_DRYRUN=1, OVERALL_BOOK_SECRET,
#      IBKR_NO_PUSH_REPORT=1, IBKR_ALERT_WEBHOOK, IBKR_RECONCILE_HEARTBEAT_URL.
# Exit: 0 reconciled | 1 problem | 2 gateway/config | 3 nothing to do
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${IBKR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
mkdir -p "${REPO_ROOT}/logs"
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${REPO_ROOT}/logs/ibkr_live_reconcile.log"; }
cd "${REPO_ROOT}"
log "starting IBKR LIVE reconcile (after the close)"
ARGS=()
[ -n "${IBKR_ACCOUNTS_CONFIG:-}" ] && ARGS+=(--config "${IBKR_ACCOUNTS_CONFIG}")
[ "${IBKR_RECONCILE_DRYRUN:-0}" = "1" ] && ARGS+=(--dry-run)
"${PYTHON}" scripts/ibkr_reconcile_accounts.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | while IFS= read -r l; do log "$l"; done
RC=${PIPESTATUS[0]}
log "done (reconcile exit ${RC})"
exit "${RC}"
