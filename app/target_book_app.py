"""Target Book (IBKR) — a friendly viewer for the signed Option-C target book.

Selected from the sidebar **Application** radio ("📋 Target Book (IBKR)").  It
renders the artifact that ``scripts/publish_target_book.py`` produces and the
executor (``scripts/ibkr_execute_book.py``) trades — the same JSON, but shown as
a human-readable allocation instead of raw text:

  * signature status (verified / unsigned / MISMATCH) and freshness,
  * the target weights mapped to the instruments actually traded on IBKR
    (BTC → IBIT), with a donut, a per-instrument table, and $/share sizing for a
    portfolio value you choose,
  * the per-asset action rows and the raw signed JSON (with a download button).

It reads the committed ``data/overall/target_book.json`` (what the publish
workflow writes to the branch / deploys with the app).  You can also generate a
fresh **live preview** on demand, which runs the same engine path the publisher
uses so the preview matches what a publish right now would emit.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import overall_core as ov                 # noqa: E402
import ticker_config                       # noqa: E402
import target_book as tb                   # noqa: E402
import freshness as fr                     # noqa: E402
import ibkr_symbols as sym                 # noqa: E402

BOOK_PATH = _REPO_ROOT / "data" / "overall" / "target_book.json"
BOOK_PATH_LIVE = _REPO_ROOT / "data" / "overall" / "target_book_live.json"

try:
    st.set_page_config(page_title="Target Book (IBKR)", page_icon="📋",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass


# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "TARGETBOOK", "EXECUTEDBOOK"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)", "GDXM": "⛏️  Gold Miners (GDX·NUGT)", "DAILYAUDIT": "🕵️  Daily Audit",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book (IBKR)"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "TARGETBOOK"
with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.caption("_The signed target book is the allocation the IBKR paper "
               "executor trades. See IBKR_PAPER_TRADING.md for the full flow._")


def _conf(*keys: str) -> str | None:
    """First non-empty value for any of *keys* from Streamlit secrets or the
    environment (secrets win, checked in the order given)."""
    for k in keys:
        try:
            s = st.secrets.get(k)
            if s:
                return str(s)
        except Exception:
            pass
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


def _secret() -> str | None:
    """Shared HMAC secret, from Streamlit secrets or the environment (either may
    be absent — then the book is shown but not cryptographically verified)."""
    return _conf("OVERALL_BOOK_SECRET")


# ── publish button (dispatches the GitHub publish workflow) ───────────────────
PUBLISH_WORKFLOW = "publish-target-book.yml"
_FALLBACK_REPO = "yoursprerit/btc-range-model"


def _github_repo() -> str:
    """owner/repo to dispatch against: GITHUB_REPO secret/env if set, else the
    origin remote of this checkout, else the canonical repo."""
    conf = _conf("GITHUB_REPO")
    if conf:
        return conf
    try:
        text = (_REPO_ROOT / ".git" / "config").read_text()
        m = re.search(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?\s*$",
                      text, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return _FALLBACK_REPO


def trigger_publish_workflow(repo: str, ref: str, token: str) -> tuple[bool, str]:
    """POST a workflow_dispatch for the publish Action. Returns (ok, message).

    The Action runs the full engine on fresh data, HMAC-signs the books and
    commits target_book(_live).json back to *ref* — i.e. exactly what the daily
    scheduled publish does, just on demand.
    """
    import requests
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{PUBLISH_WORKFLOW}/dispatches"
    try:
        r = requests.post(
            url, json={"ref": ref}, timeout=15,
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})
    except Exception as e:
        return False, f"Could not reach the GitHub API: {e}"
    if r.status_code == 204:
        return True, (f"Publish workflow dispatched on `{repo}@{ref}`. The engine "
                      "run takes ~5–15 min; the fresh signed book is then committed "
                      "to the branch and appears here after the app reloads it.")
    detail = ""
    try:
        msg = r.json().get("message")
        if msg:
            detail = f" — {msg}"
    except Exception:
        pass
    hint = {401: " (token invalid/expired?)",
            403: " (token lacks the `actions: write` / workflow scope?)",
            404: " (repo, workflow file or ref not found — or the token cannot see the repo)",
            422: f" (ref `{ref}` has no {PUBLISH_WORKFLOW}?)"}.get(r.status_code, "")
    return False, f"GitHub API returned {r.status_code}{hint}{detail}"


def _render_publish_button() -> None:
    repo = _github_repo()
    ref = _conf("GITHUB_PUBLISH_REF") or "main"
    token = _conf("GITHUB_TOKEN", "GH_TOKEN")
    actions_url = f"https://github.com/{repo}/actions/workflows/{PUBLISH_WORKFLOW}"
    cols = st.columns([1, 2])
    clicked = cols[0].button(
        "🚀 Publish new target book", type="primary", use_container_width=True,
        disabled=not token,
        help="Runs the **Publish target book (IBKR Option C)** GitHub Action now: "
             "the engine recomputes today's allocation from the freshest data, "
             "signs it, and commits the new target_book(_live).json to "
             f"`{repo}@{ref}` — same path as the daily scheduled publish.")
    if not token:
        cols[1].caption(
            "_Add a `GITHUB_TOKEN` (fine-grained PAT with **Actions: write** on "
            f"this repo) to Streamlit secrets to enable this button — or run the "
            f"[workflow manually]({actions_url})._")
        return
    if clicked:
        with st.spinner("Dispatching the publish workflow…"):
            ok, msg = trigger_publish_workflow(repo, ref, token)
        if ok:
            st.success(msg)
            st.caption(f"_Watch progress on the [Actions page]({actions_url})._")
        else:
            st.error(f"Dispatch failed: {msg}")
            st.caption(f"_Fallback: run the [workflow manually]({actions_url}) or "
                       "`python scripts/publish_target_book.py` locally._")


# ── reallocation helpers (UI-free, unit-tested) ───────────────────────────────
# The include/exclude maths lives in overall_core so the Overall app's Live tab
# and this viewer behave identically (single source of truth).
adjust_for_selection = ov.adjust_for_selection


def build_adjusted_payload(payload: dict, adj_weights: dict, adj_cash: float) -> dict:
    """Rebuild a target-book payload from the user's selection, preserving the
    book's identity (as-of, profile, generation time, paper/live mode) and the
    kept names' prices so freshness and downstream execution still work."""
    adj = tb.build_payload(
        as_of=payload.get("as_of", ""), profile=payload.get("profile", ""),
        weights=adj_weights, cash_weight=adj_cash,
        exec_price={k: v for k, v in (payload.get("exec_price") or {}).items()
                    if k in adj_weights},
        actions=payload.get("actions"),
        generated_at_utc=payload.get("generated_at_utc"),
        book_mode=payload.get("book_mode", "paper"))
    if payload.get("signal_audit"):        # carry the audit verdict through
        adj["signal_audit"] = dict(payload["signal_audit"])
    return adj


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color}22;color:{color};font-weight:700;"
            f"padding:3px 11px;border-radius:999px;font-size:13px'>{text}</span>")


def _render_book(payload: dict, *, source: str) -> None:
    secret = _secret()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok_sig, sig_why = tb.verify_signature(payload, secret)
    ok_val, val_why = tb.validate(payload, today)

    is_live = payload.get("book_mode") == "live"
    idle_label = "SATA" if is_live else "Cash"       # where idle capital parks
    prices = payload.get("exec_price", {}) or {}
    cash = float(payload.get("cash_weight", 0.0) or 0.0)
    # split the idle bucket out of the weights: for live it's the SATA position,
    # for paper it's cash_weight (SATA is never a paper key).
    all_w = {k: float(v) for k, v in (payload.get("weights") or {}).items()}
    idle_base = all_w.pop("SATA", 0.0) if is_live else cash
    risk_w = all_w                                    # risk positions only

    # ── header + status badges ────────────────────────────────────────────────
    st.caption(f"Source: {source}")
    c = st.columns(2)
    c[0].metric("As-of (signal bar)", payload.get("as_of", "—"))
    c[1].metric("Risk profile", payload.get("profile", "—"))
    gen = payload.get("generated_at_utc", "—")

    acct_txt = "🔴 LIVE book — idle → SATA" if is_live else "🧪 PAPER book — idle → cash"
    acct_col = "#dc2626" if is_live else "#0ea5e9"
    signed = bool(payload.get("signature"))
    if not secret:
        sig_color = "#64748b"; sig_txt = "🔒 signed (no key to verify)" if signed else "🔓 unsigned"
    elif ok_sig:
        sig_color = "#16a34a"; sig_txt = "✅ signature verified"
    else:
        sig_color = "#dc2626"; sig_txt = "⛔ signature MISMATCH"
    val_color = "#16a34a" if ok_val else "#d97706"

    # ── daily signal-freshness audit badge (like the signature badge) ─────────
    # The publisher stamps the audit verdict INTO the signed book
    # (``signal_audit``); older books lack the stamp, so fall back to the
    # committed repo audit trail when it refers to THIS exact publish.
    aud = payload.get("signal_audit")
    aud_note = ""
    if not aud:
        _trail = fr.load_daily_audit() or {}
        _tb_info = _trail.get("target_book") or {}
        if (_tb_info.get("generated_at_utc")
                and _tb_info.get("generated_at_utc") == payload.get("generated_at_utc")):
            aud = _trail.get("audit") or {}
            aud_note = " (repo audit trail)"
    if not aud:
        aud_color = "#64748b"; aud_txt = "🕵️ daily audit not stamped (pre-audit book)"
    elif aud.get("passed"):
        aud_color = "#16a34a"
        aud_txt = f"🕵️ daily audit passed{aud_note}"
    else:
        aud_color = "#dc2626"
        aud_txt = f"🚨 daily audit FAILED{aud_note}"

    st.markdown(
        _badge(acct_txt, acct_col) + "  " +
        _badge(sig_txt, sig_color) + "  " +
        _badge(aud_txt, aud_color) + "  " +
        _badge(("🟢 " if ok_val else "🟡 ") + val_why, val_color) + "  " +
        _badge(f"generated {gen} UTC", "#0ea5e9"),
        unsafe_allow_html=True)
    if secret and not ok_sig:
        st.error(f"Signature check failed — {sig_why}. This book would be REFUSED "
                 "by the executor. Do not trade it.")
    if aud and not aud.get("passed"):
        _stale = ", ".join(aud.get("stale_apps") or []) or "unknown"
        st.error(f"🚨 The daily signal-freshness audit FAILED when this book was "
                 f"computed — stale signal apps: **{_stale}** "
                 f"(checked {fr.fmt_ct(aud.get('checked_at_utc'))}). The executor "
                 f"refuses audit-failed books. Do not trade it.")
    st.markdown("")

    if not risk_w and idle_base <= 1e-9:
        st.info("This book is **fully in cash** — no deployed positions today.")
        _raw_expander(payload, secret)
        return

    # ── portfolio-value sizer ─────────────────────────────────────────────────
    nav = st.number_input("Portfolio value ($) for $/share sizing",
                          min_value=1000, value=100_000, step=10_000, format="%d")

    st.markdown("#### Target allocation")
    st.caption(f"Untick **Include** to drop a position — its weight moves to "
               f"**{idle_label}** (it is *not* redistributed to the other assets).")

    base_rows = []
    for k in sorted(risk_w, key=lambda k: -risk_w[k]):
        meta = ov.ASSET_META.get(k, {})
        base_rows.append(dict(
            Include=True, Signal=k, IBKR=(sym.trade_symbol(k) or k),
            Instrument=f"{ov.KIND_EMOJI.get(meta.get('kind', 'core'), '')} "
                       f"{meta.get('name', k)}".strip(),
            Weight=risk_w[k] * 100, Price=(prices.get(k) or float('nan'))))
    base_df = pd.DataFrame(base_rows)

    # Per-row Include checkbox = the select/unselect control. Keyed by book_mode so
    # paper and live keep independent selections; state persists across reruns.
    edited = st.data_editor(
        base_df, hide_index=True, use_container_width=True,
        key=f"tb_selection_{'live' if is_live else 'paper'}",
        disabled=["Signal", "IBKR", "Instrument", "Weight", "Price"],
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", default=True),
            "Weight": st.column_config.NumberColumn("Weight %", format="%.1f%%"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f")})

    included = set(edited.loc[edited["Include"], "Signal"]) if len(base_df) else set()
    excluded = [k for k in risk_w if k not in included]
    moved = sum(risk_w[k] for k in excluded)
    adj_risk = {k: v for k, v in risk_w.items() if k in included}
    idle_after = idle_base + moved                    # excluded weight → idle bucket
    risk_total = sum(adj_risk.values())

    m = st.columns(3)
    m[0].metric("Risk assets", f"{risk_total * 100:.0f}%",
                delta=(f"−{moved * 100:.0f}%" if moved > 1e-9 else None),
                delta_color="inverse")
    m[1].metric(idle_label, f"{idle_after * 100:.0f}%", f"${idle_after * nav:,.0f}",
                delta_color="off")
    m[2].metric("Positions", f"{len(included)}/{len(risk_w)}")
    if excluded:
        st.info(f"Excluded **{', '.join(excluded)}** → moved "
                f"**{moved * 100:.1f}%** (${moved * nav:,.0f}) to {idle_label}.")

    # deployable positions = risk + (SATA as a real position on live)
    deploy_w = dict(adj_risk)
    if is_live and idle_after > 1e-9:
        deploy_w["SATA"] = idle_after

    left, right = st.columns([3, 2])
    with left:
        size_rows = []
        for k in sorted(deploy_w, key=lambda k: -deploy_w[k]):
            w = deploy_w[k]; px = prices.get(k)
            size_rows.append(dict(
                IBKR=(sym.trade_symbol(k) or k), Weight=f"{w * 100:.1f}%",
                Value=f"${w * nav:,.0f}",
                Shares=(f"{(w * nav / px):,.0f}" if px else "—")))
        if size_rows:
            st.markdown(f"**Deployed positions — $ / shares** "
                        f"{'(incl. SATA)' if is_live else ''}")
            st.dataframe(pd.DataFrame(size_rows), hide_index=True,
                         use_container_width=True)
        else:
            st.warning(f"All positions unticked — the whole book is {idle_label}.")
    with right:
        keys_sorted = sorted(deploy_w, key=lambda k: -deploy_w[k])
        labels = [(sym.trade_symbol(k) or k) for k in keys_sorted]
        vals = [deploy_w[k] * 100 for k in keys_sorted]
        if not is_live and idle_after > 0.001:        # paper cash slice
            labels += ["CASH"]; vals += [idle_after * 100]
        fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.55,
                               sort=False, textinfo="label+percent"))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── action rows (annotated with the user's exclusions) ─────────────────────
    actions = payload.get("actions") or []
    if actions:
        st.markdown("#### Per-asset actions")
        adf = pd.DataFrame([
            dict(Signal=a.get("key"),
                 Action=(f"EXCLUDED → {idle_label}" if a.get("key") in excluded
                         else a.get("action")),
                 Decision=a.get("decision"),
                 Target=(f"{(a.get('target') or 0) * 100:.1f}%"),
                 Held=("yes" if a.get("in_pos") else "—"),
                 Priority=(f"{a['priority']:.2f}" if a.get("priority") is not None else "—"))
            for a in actions])
        st.dataframe(adf, hide_index=True, use_container_width=True)

    # ── downloads: original + (if edited) an adjusted, re-signed book ──────────
    # live keeps idle in the SATA weight (cash_weight 0); paper keeps it as cash.
    if is_live:
        dl_weights = dict(adj_risk)
        if idle_after > 1e-9:
            dl_weights["SATA"] = idle_after
        dl_cash = 0.0
    else:
        dl_weights, dl_cash = adj_risk, idle_after
    _downloads(payload, dl_weights, dl_cash, excluded, secret)


def _downloads(payload: dict, adj_weights: dict, adj_cash: float,
               excluded: list, secret) -> None:
    """Original book download + a user-adjusted book that keeps unticked assets'
    weight in cash. The adjusted book is re-signed when a secret is configured so
    the executor still accepts it; otherwise it is unsigned (and flagged)."""
    orig_text = (json.dumps(payload, indent=1) if payload.get("signature")
                 else tb.dumps(payload, secret))
    with st.expander("Raw signed JSON (as published)"):
        st.code(orig_text, language="json")

    cols = st.columns(2)
    cols[0].download_button("⬇️ Original book", data=orig_text,
                            file_name="target_book.json", mime="application/json",
                            use_container_width=True)

    if not excluded:
        cols[1].caption("_No exclusions — the adjusted book equals the original._")
        return

    # rebuild the payload with the user's selection; keep as_of / generated_at /
    # profile / exec_price (for the kept names) so freshness + identity carry over.
    adj_payload = build_adjusted_payload(payload, adj_weights, adj_cash)
    adj_text = tb.dumps(adj_payload, secret)
    cols[1].download_button(
        f"⬇️ Adjusted book ({'signed' if secret else 'UNSIGNED'})",
        data=adj_text, file_name="target_book_adjusted.json",
        mime="application/json", type="primary", use_container_width=True)
    if not secret:
        st.caption("⚠️ No secret configured, so the adjusted book is **unsigned** — "
                   "the executor will reject it unless run without a secret / "
                   "`--require-signature`. Configure `OVERALL_BOOK_SECRET` to sign it.")


@st.cache_data(ttl=900, show_spinner="Running the engine for a live preview (~30–90s)…")
def _live_payload(profile: str, bucket: str) -> dict:
    """Compute a fresh book via the exact publisher path (reused, no drift):
    run the universe once, AUDIT its signal freshness, then build the book from
    the same audited results and stamp the verdict in — so the preview carries
    the same 🕵️ daily-audit badge as a published book."""
    from ibkr_rebalance import compute_target_book
    results = ov.run_universe()
    audit = fr.audit_universe(results, parent_order=ov.PARENT_KEYS)
    book = compute_target_book(profile, results=results)
    payload = tb.build_payload(
        as_of=book.as_of, profile=profile, weights=book.weights,
        cash_weight=book.cash_weight, exec_price=book.exec_price, actions=book.actions)
    payload["signal_audit"] = dict(passed=bool(audit["passed"]),
                                   checked_at_utc=audit["checked_at_utc"],
                                   stale_apps=list(audit["stale_apps"]))
    return payload


def _bucket() -> str:
    now = pd.Timestamp.utcnow()
    return f"{now.date()}-{now.hour}-{now.minute // 30}"


# ══════════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════════
st.title("📋 Target Book (IBKR)")
st.caption("The signed allocation the IBKR executor trades — mapped to the "
           "instruments actually traded (BTC → IBIT). **Paper** parks idle capital "
           "as cash; **Live** parks it in a SATA position.")

_src = st.radio("Book source", ["📦 Published artifact", "🔬 Live preview"],
                horizontal=True,
                help="Published = the committed target_book(_live).json. Live "
                     "preview = compute it now via the same engine the publisher "
                     "uses (unsigned unless a secret is configured).")

if _src.startswith("📦"):
    _render_publish_button()
    st.markdown("")
    # Paper / Live account selector — Live appears once a live book is published.
    _avail = [("Paper", BOOK_PATH)] + (
        [("Live", BOOK_PATH_LIVE)] if BOOK_PATH_LIVE.exists() else [])
    if len(_avail) > 1:
        _pick = st.radio("Account", [n for n, _ in _avail], horizontal=True,
                         key="tb_account",
                         help="Paper (idle → cash) and Live (idle → SATA) are "
                              "published as separate books; the executor picks the "
                              "one matching its --account-mode.")
        _path = dict(_avail)[_pick]
    else:
        _path = BOOK_PATH
    if _path.exists():
        try:
            payload = tb.loads(_path.read_text())
            # freshness caption + 🕵️ Daily Audit bookkeeping (shared helpers so
            # the publish time shown here matches the Daily Audit tab)
            try:
                import freshness as _fr
                st.caption(
                    f"📋 Target Book generated & published "
                    f"**{_fr.fmt_ct(payload.get('generated_at_utc'))}** from the "
                    f"**{payload.get('as_of', '—')}** signal bar · 🔄 page data "
                    f"refreshed **{_fr.fmt_ct(_fr.now_utc(), seconds=True)}**")
                _fr.record_refresh("TARGETBOOK", kind="targetbook",
                                   app_label="📋 Target Book (IBKR)",
                                   as_of=str(payload.get("as_of", "")),
                                   generated_at_utc=str(payload.get("generated_at_utc", "")),
                                   book_mode=str(payload.get("book_mode", "paper")),
                                   profile=str(payload.get("profile", "")))
            except Exception:
                pass
            _render_book(payload, source=f"`{_path.relative_to(_REPO_ROOT)}`")
        except Exception as e:
            st.error(f"Could not read the published book: {e}")
    else:
        st.warning(f"No published target book found at "
                   f"`{_path.relative_to(_REPO_ROOT)}`.")
        st.markdown(
            "Publish one with the **Publish target book (IBKR Option C)** GitHub "
            "Action (or `python scripts/publish_target_book.py`), then it appears "
            "here. Meanwhile you can use **🔬 Live preview** above.")
else:
    profile = st.session_state.get("overall_risk_profile", ov.DEFAULT_PROFILE)
    if profile not in ov.RISK_PROFILES:
        profile = ov.DEFAULT_PROFILE
    st.caption(f"Live preview uses the **{profile}** profile "
               "(switch it on the 🧭 Overall Trading → Live tab).")
    if st.button("▶️ Generate live preview", type="primary"):
        st.session_state["_tb_preview_ready"] = True
    if st.session_state.get("_tb_preview_ready"):
        try:
            payload = _live_payload(profile, _bucket())
            secret = _secret()
            if secret:                       # sign so the preview verifies too
                payload = tb.sign(payload, secret)
            _render_book(payload, source="live engine run")
        except Exception as e:
            st.error(f"Live preview failed: {e}")
