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
import ibkr_symbols as sym                 # noqa: E402

BOOK_PATH = _REPO_ROOT / "data" / "overall" / "target_book.json"

try:
    st.set_page_config(page_title="Target Book (IBKR)", page_icon="📋",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass


# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = ["OVERALL", "BTC", "GLDM"] + ticker_config.APP_KEYS + ["TARGETBOOK"]
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold (GLDM)", "TARGETBOOK": "📋  Target Book (IBKR)"}
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


def _secret() -> str | None:
    """Shared HMAC secret, from Streamlit secrets or the environment (either may
    be absent — then the book is shown but not cryptographically verified)."""
    try:
        s = st.secrets.get("OVERALL_BOOK_SECRET")
        if s:
            return str(s)
    except Exception:
        pass
    return os.environ.get("OVERALL_BOOK_SECRET")


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color}22;color:{color};font-weight:700;"
            f"padding:3px 11px;border-radius:999px;font-size:13px'>{text}</span>")


def _render_book(payload: dict, *, source: str) -> None:
    secret = _secret()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    ok_sig, sig_why = tb.verify_signature(payload, secret)
    ok_val, val_why = tb.validate(payload, today)

    weights = payload.get("weights", {}) or {}
    prices = payload.get("exec_price", {}) or {}
    cash = float(payload.get("cash_weight", 0.0) or 0.0)

    # ── header + status badges ────────────────────────────────────────────────
    st.caption(f"Source: {source}")
    c = st.columns(4)
    c[0].metric("As-of (signal bar)", payload.get("as_of", "—"))
    c[1].metric("Risk profile", payload.get("profile", "—"))
    c[2].metric("Deployed", f"{(1.0 - cash) * 100:.0f}%")
    c[3].metric("Cash", f"{cash * 100:.0f}%")
    gen = payload.get("generated_at_utc", "—")

    signed = bool(payload.get("signature"))
    if not secret:
        sig_color = "#64748b"; sig_txt = "🔒 signed (no key to verify)" if signed else "🔓 unsigned"
    elif ok_sig:
        sig_color = "#16a34a"; sig_txt = "✅ signature verified"
    else:
        sig_color = "#dc2626"; sig_txt = "⛔ signature MISMATCH"
    val_color = "#16a34a" if ok_val else "#d97706"
    st.markdown(
        _badge(sig_txt, sig_color) + "  " +
        _badge(("🟢 " if ok_val else "🟡 ") + val_why, val_color) + "  " +
        _badge(f"generated {gen} UTC", "#0ea5e9"),
        unsafe_allow_html=True)
    if secret and not ok_sig:
        st.error(f"Signature check failed — {sig_why}. This book would be REFUSED "
                 "by the executor. Do not trade it.")
    st.markdown("")

    if not weights:
        st.info("This book is **fully in cash** — no deployed positions today.")
        _raw_expander(payload, secret)
        return

    # ── portfolio-value sizer ─────────────────────────────────────────────────
    nav = st.number_input("Paper portfolio value ($) for $/share sizing",
                          min_value=1000, value=100_000, step=10_000, format="%d")

    rows = []
    for k in sorted(weights, key=lambda k: -weights[k]):
        w = float(weights[k])
        px = prices.get(k)
        symbol = sym.trade_symbol(k) or k
        meta = ov.ASSET_META.get(k, {})
        name = meta.get("name", k)
        kind = meta.get("kind", "core")
        dollars = w * nav
        shares = (dollars / px) if px else None
        rows.append(dict(
            Instrument=f"{ov.KIND_EMOJI.get(kind, '')} {name}".strip(),
            Signal=k, IBKR=symbol, Weight=w * 100,
            Price=(f"${px:,.2f}" if px else "—"),
            Value=f"${dollars:,.0f}",
            Shares=(f"{shares:,.0f}" if shares else "—")))
    df = pd.DataFrame(rows)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Target allocation")
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={"Weight": st.column_config.NumberColumn(
                "Weight %", format="%.1f%%")})
        st.caption(f"Cash (uninvested): **{cash * 100:.1f}%** · "
                   f"${cash * nav:,.0f}")
    with right:
        labels = [f"{r['IBKR']}" for r in rows] + (["CASH"] if cash > 0.001 else [])
        vals = [r["Weight"] for r in rows] + ([cash * 100] if cash > 0.001 else [])
        fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.55,
                               sort=False, textinfo="label+percent"))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── action rows ───────────────────────────────────────────────────────────
    actions = payload.get("actions") or []
    if actions:
        st.markdown("#### Per-asset actions")
        adf = pd.DataFrame([
            dict(Signal=a.get("key"), Action=a.get("action"),
                 Decision=a.get("decision"),
                 Target=(f"{(a.get('target') or 0) * 100:.1f}%"),
                 Held=("yes" if a.get("in_pos") else "—"),
                 Priority=(f"{a['priority']:.2f}" if a.get("priority") is not None else "—"))
            for a in actions])
        st.dataframe(adf, hide_index=True, use_container_width=True)

    _raw_expander(payload, secret)


def _raw_expander(payload: dict, secret) -> None:
    text = tb.dumps(payload, secret) if payload.get("signature") is None else json.dumps(
        payload, indent=1)
    with st.expander("Raw signed JSON"):
        st.code(text, language="json")
    st.download_button("⬇️ Download target_book.json", data=text,
                       file_name="target_book.json", mime="application/json")


@st.cache_data(ttl=900, show_spinner="Running the engine for a live preview (~30–90s)…")
def _live_payload(profile: str, bucket: str) -> dict:
    """Compute a fresh book via the exact publisher path (reused, no drift)."""
    from ibkr_rebalance import compute_target_book
    book = compute_target_book(profile)
    return tb.build_payload(
        as_of=book.as_of, profile=profile, weights=book.weights,
        cash_weight=book.cash_weight, exec_price=book.exec_price, actions=book.actions)


def _bucket() -> str:
    now = pd.Timestamp.utcnow()
    return f"{now.date()}-{now.hour}-{now.minute // 30}"


# ══════════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════════
st.title("📋 Target Book (IBKR)")
st.caption("The signed allocation the IBKR **paper** executor trades — mapped to "
           "the instruments actually traded (BTC → IBIT).")

_src = st.radio("Book source", ["📦 Published artifact", "🔬 Live preview"],
                horizontal=True,
                help="Published = the committed data/overall/target_book.json. "
                     "Live preview = compute it now via the same engine the "
                     "publisher uses (unsigned unless a secret is configured).")

if _src.startswith("📦"):
    if BOOK_PATH.exists():
        try:
            payload = tb.loads(BOOK_PATH.read_text())
            _render_book(payload, source=f"`{BOOK_PATH.relative_to(_REPO_ROOT)}`")
        except Exception as e:
            st.error(f"Could not read the published book: {e}")
    else:
        st.warning("No published target book found at "
                   f"`{BOOK_PATH.relative_to(_REPO_ROOT)}`.")
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
