#!/usr/bin/env bash
# Daily IBKR paper executor (Option C) for a cloud VM / cron — the Linux twin of
# scripts/ibkr_execute_daily.ps1.
#
# Pulls the latest published target book from the branch, then runs the
# lightweight executor to rebalance the IBKR *paper* account to match it. The
# model never runs here — this host only consumes the pre-computed book.
#
# The executor itself guards against weekends / holidays / stale books and only
# trades a paper (DU…) account, so a stray or early run is a safe no-op.
#
# Example crontab (weekdays 09:45 America/New_York — a few minutes after the US
# open, after the cloud publisher has committed the morning's book). Set CRON_TZ
# so the schedule tracks Eastern regardless of the host clock:
#
#     CRON_TZ=America/New_York
#     45 9 * * 1-5  /path/to/repo/scripts/ibkr_execute_daily.sh >> /path/to/repo/logs/ibkr_cron.log 2>&1
#
# Env overrides (see also the flags below):
#   IBKR_PYTHON   interpreter to use (default: <repo>/.venv/bin/python)
#   IBKR_BRANCH   branch carrying target_book.json (default: main)
#   IBKR_BOOK     path to the book (default: <repo>/data/overall/target_book.json)
#   IBKR_BAND     no-trade band as a fraction of net-liq (default: 0.01)
#   IBKR_HOST     IB Gateway host (default: 127.0.0.1)
#   IBKR_PORT     IB Gateway API port (default: 4002; the Docker image often
#                 remaps paper to 4004 — set this to match)
#   IBKR_EXTRA    extra flags passed through (e.g. "--fractional")
#   IBKR_NO_PULL  set to 1 to skip the git pull (use the book already on disk)
#
# OVERALL_BOOK_SECRET must be exported (same value used to publish) so the book's
# signature verifies before any order is placed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${IBKR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
BRANCH="${IBKR_BRANCH:-main}"
ACCOUNT_MODE="${IBKR_ACCOUNT_MODE:-paper}"
# Pick the right published book by mode: live parks idle capital in SATA, paper
# holds it as cash. Override with IBKR_BOOK.
if [ "${ACCOUNT_MODE}" = "live" ]; then
  DEFAULT_BOOK="${REPO_ROOT}/data/overall/target_book_live.json"
else
  DEFAULT_BOOK="${REPO_ROOT}/data/overall/target_book.json"
fi
BOOK="${IBKR_BOOK:-${DEFAULT_BOOK}}"
BAND="${IBKR_BAND:-0.01}"
HOST="${IBKR_HOST:-127.0.0.1}"
PORT="${IBKR_PORT:-4002}"
EXTRA="${IBKR_EXTRA:-}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

cd "${REPO_ROOT}"
log "──────────────────────────────────────────────"
log "starting IBKR paper executor (branch=${BRANCH} host=${HOST} port=${PORT} band=${BAND})"

if [ "${IBKR_NO_PULL:-0}" != "1" ]; then
  log "fetching latest book from origin/${BRANCH}"
  # Fast-forward only: take the newest committed book without clobbering local work.
  git fetch --quiet origin "${BRANCH}"
  git merge --ff-only "origin/${BRANCH}" || log "ff-merge skipped (local diverged?) — using on-disk book"
fi

# For live the wrapper passes --confirm-live and the live safety limits from the
# env; set IBKR_ACCOUNT_MODE=live only in the LIVE env file
# (deploy/systemd/executor-live.env), pointing IBKR_PORT at 4003.
ARGS=(--file "${BOOK}" --execute --band "${BAND}" --host "${HOST}" --port "${PORT}"
      --account-mode "${ACCOUNT_MODE}")
if [ "${ACCOUNT_MODE}" = "live" ]; then
  ARGS+=(--confirm-live)
  [ -n "${IBKR_EXPECTED_ACCOUNT:-}" ]   && ARGS+=(--expected-account "${IBKR_EXPECTED_ACCOUNT}")
  [ -n "${IBKR_MAX_DEPLOY_FRAC:-}" ]    && ARGS+=(--max-deploy-frac "${IBKR_MAX_DEPLOY_FRAC}")
  [ -n "${IBKR_MAX_ORDER_NOTIONAL:-}" ] && ARGS+=(--max-order-notional "${IBKR_MAX_ORDER_NOTIONAL}")
fi
[ -n "${IBKR_KILL_SWITCH_FILE:-}" ] && ARGS+=(--kill-switch-file "${IBKR_KILL_SWITCH_FILE}")
# shellcheck disable=SC2206
[ -n "${EXTRA}" ] && ARGS+=(${EXTRA})

log "account mode: ${ACCOUNT_MODE}"

# --execute places orders; the executor's own guards decide if it actually trades.
"${PYTHON}" scripts/ibkr_execute_book.py "${ARGS[@]}"

# Publish the execution report back so the cloud app's "Executed Book" tab shows
# it. Requires git WRITE credentials on this host (deploy key / token). On push
# failure we reset to origin so the branch never diverges and tomorrow's
# ff-merge of the fresh target book still works. Set IBKR_NO_PUSH_REPORT=1 to skip.
if [ "${ACCOUNT_MODE}" = "live" ]; then
  REPORT="${REPO_ROOT}/data/overall/executed_book_live.json"
else
  REPORT="${REPO_ROOT}/data/overall/executed_book.json"
fi
if [ "${IBKR_NO_PUSH_REPORT:-0}" != "1" ] && [ -f "${REPORT}" ]; then
  git add "${REPORT}"
  if git diff --cached --quiet -- "${REPORT}"; then
    log "no change to executed_book.json — nothing to publish"
  else
    git -c user.name="ibkr-executor" -c user.email="executor@localhost" \
      commit -q -m "chore(ibkr): execution report $(date -u +%F)"
    if git push origin "HEAD:${BRANCH}"; then
      log "published execution report to origin/${BRANCH}"
    else
      log "WARN: could not push execution report (this host needs git write access) — rolling back"
      git reset --hard "origin/${BRANCH}" >/dev/null 2>&1 || true
    fi
  fi
fi

log "done"
