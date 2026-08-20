"""🤖 AI Assistant — ask questions about this app's data, calculations and design.

A chat page grounded in the repository itself.  Two things make the answers
checkable rather than plausible:

  * every turn carries a **state pack** — the same committed artifacts the
    📋 Target Book, ✅ Executed Book, 🩺 Strategy Health, 🕵️ Daily Audit and
    🧭 Overall Backtesting tabs render — so "what is the book holding?" is
    answered from the published JSON, not from memory;
  * the model has **read-only tools** over the working tree (grep, file read,
    JSON drill-down, dataframe head/tail/describe), so "*why* is it holding
    that?" is answered from `OVERALL_STRATEGY.md`, the `*_EVAL.md` record or
    the actual `app/*.py` calculation — with the path cited.

Nothing on this page can write: the sandbox in ``assistant_core`` confines
every read to the repo and refuses anything that looks like a credential.

The model picker lists what this API key is really entitled to (``GET
/v1/models``), most capable first, with each model's price shown.  The
*cheapest* entitled model is selected by default — most questions here are
look-ups the state pack already answers — and a stronger one is one click away
for questions that need several look-ups chained together.

Follow-up questions stay in the same session: the whole thread is replayed to
the API each turn, so "and how does that compare to GLDM?" works.  **🆕 New
chat** drops the thread and starts fresh.

Configuration
-------------
Set ``ANTHROPIC_API_KEY`` in the environment, or in Streamlit secrets::

    # .streamlit/secrets.toml   (git-ignored — never commit a key)
    ANTHROPIC_API_KEY = "sk-ant-…"
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ticker_config                        # noqa: E402
import freshness as fr                      # noqa: E402
import assistant_core as ac                 # noqa: E402

try:
    st.set_page_config(page_title="AI Assistant", page_icon="🤖",
                       layout="wide", initial_sidebar_state="expanded")
except Exception:
    pass

# ── sidebar Application selector (same widget/key as every other app) ─────────
_ALL_APPS = (["OVERALL", "BTC", "GLDM", "GDXM"] + ticker_config.APP_KEYS
             + ["DAILYAUDIT", "HEALTH", "TARGETBOOK", "EXECUTEDBOOK", "ASSISTANT"])
_APP_LABELS = {"OVERALL": "🧭  Overall Trading", "BTC": "₿  Bitcoin (BTC)",
               "GLDM": "🥇  Gold Trend (GLDM·UGL)",
               "GDXM": "⛏️  Gold Miners (GDX·NUGT)", "DAILYAUDIT": "🕵️  Daily Audit",
               "HEALTH": "🩺  Strategy Health",
               "TARGETBOOK": "📋  Target Book (IBKR)",
               "EXECUTEDBOOK": "✅  Executed Book (IBKR)",
               "ASSISTANT": "🤖  AI Assistant"}
for _k, _c in ticker_config.CONFIGS.items():
    _APP_LABELS[_k] = f"{_c.emoji}  {_c.key} · {_c.name.split('(')[0].strip()[:22]}"
if st.session_state.get("gldm_active_app") not in _ALL_APPS:
    st.session_state["gldm_active_app"] = "ASSISTANT"

_HISTORY_KEY = "assistant_history"      # what the API sees (content blocks)
_TRANSCRIPT_KEY = "assistant_transcript"  # what the user sees (rendered turns)


@st.cache_data(ttl=900, show_spinner=False)
def _discover(api_key_fingerprint: str, _key: str):
    """Entitled models for this key.  Cached on a fingerprint rather than the
    key itself so the secret never becomes part of a cache key that might be
    displayed; the key is passed separately as an ignored argument."""
    specs, err = ac.discover_models(_key)
    return [(s.model_id, s.label, s.blurb) for s in specs], err


@st.cache_data(ttl=300, show_spinner=False)
def _system_prompt(artifact_mtimes: tuple):
    """The grounded system prompt, rebuilt when any artifact changes on disk.

    Keyed on the artifacts' mtimes so a nightly publish invalidates it, and
    stable in between — which is exactly what the API's prompt cache needs.
    """
    return ac.build_system_prompt()


def _artifact_mtimes() -> tuple:
    paths = sorted((_REPO_ROOT / "data" / "overall").glob("*.json"))
    paths += sorted(_REPO_ROOT.glob("data/*/backtest_results.json"))
    out = []
    for p in paths:
        try:
            out.append((p.name, round(p.stat().st_mtime, 3)))
        except OSError:
            pass
    return tuple(out)


_API_KEY = ac.resolve_api_key()

with st.sidebar:
    st.radio("**Application**", options=_ALL_APPS,
             format_func=lambda x: _APP_LABELS.get(x, x), key="gldm_active_app")
    st.markdown("---")
    st.markdown("#### 🤖 Model")

    if _API_KEY:
        _choices, _discover_err = _discover(f"…{_API_KEY[-6:]}", _API_KEY)
    else:
        _choices, _discover_err = [], None
        _fallback = sorted(ac.MODEL_CATALOG.values(), key=lambda s: s.rank)
        _choices = [(s.model_id, s.label, s.blurb) for s in _fallback]

    _ids = [c[0] for c in _choices]
    _labels = {c[0]: c[1] for c in _choices}
    _blurbs = {c[0]: c[2] for c in _choices}
    _specs = [ac.spec_for(i) for i in _ids]
    _default = ac.default_model(_specs)          # the cheapest entitled model
    _cheapest = ac.cheapest_model(_specs)
    _capable = ac.most_capable_model(_specs)
    if st.session_state.get("assistant_model") not in _ids and _ids:
        st.session_state["assistant_model"] = _default

    def _model_label(mid: str) -> str:
        """Name + why you would pick it + what a million input tokens costs."""
        spec = ac.spec_for(mid)
        tag = ("  💲 cheapest" if mid == _cheapest
               else "  🧠 most capable" if mid == _capable else "")
        # \$ escaped: Streamlit renders these labels as markdown, and a bare
        # "$5/$25" is parsed as LaTeX math — the dollars vanish and the text
        # reflows.
        price = (f"  ·  \\${spec.input_price:g}/\\${spec.output_price:g} per Mtok"
                 if spec.priced else "")
        return f"{_labels.get(mid, mid)}{tag}{price}"

    _model_id = st.radio(
        "Anthropic model", options=_ids, format_func=_model_label,
        key="assistant_model",
        help="Models your Anthropic subscription can use, most capable first. "
             "Prices are USD per million input/output tokens. The cheapest one "
             "is selected by default — switch to a stronger model for questions "
             "that need several look-ups chained together.")
    st.caption(f"_{_blurbs.get(_model_id, '')}_")
    st.caption(f"`{_model_id}`")
    if _discover_err:
        st.caption(f"⚠️ _Could not list your models ({_discover_err}) — showing "
                   "the built-in list._")

    _spec = ac.spec_for(_model_id)
    _show_reasoning = st.toggle(
        "Show reasoning", value=False,
        disabled=not _spec.supports_reasoning,
        help=("A summary of the model's reasoning is shown above each answer."
              if _spec.supports_reasoning else
              f"{_spec.label} does not expose reasoning."))

    st.markdown("---")
    st.markdown("#### 💬 Session")
    _turns = len(st.session_state.get(_TRANSCRIPT_KEY, []))
    # The sidebar runs BEFORE this script run appends the current question, so
    # reading the transcript here is always one turn stale.  Hold the slot and
    # refresh it after the answer lands (see _render_session_caption below).
    _session_slot = st.empty()
    # Starting a new chat drops the API-side history AND the visible transcript,
    # so the next question is billed against a fresh (cached) prefix instead of
    # dragging an unrelated thread along.
    if st.button("🆕 New chat", use_container_width=True, type="primary",
                 disabled=not _turns,
                 help="Forget this conversation and start a fresh session. "
                      "The app data and documentation context is unchanged."):
        st.session_state[_HISTORY_KEY] = []
        st.session_state[_TRANSCRIPT_KEY] = []
        st.session_state["assistant_session_n"] = (
            st.session_state.get("assistant_session_n", 1) + 1)
        st.rerun()
    if st.button("♻️ Reload app data", use_container_width=True,
                 help="Re-read the committed artifacts into the assistant's context."):
        _system_prompt.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("_Answers are grounded in the committed artifacts and the repo's "
               "own documentation — the assistant reads files, it never writes "
               "and never places an order._")

def _render_session_caption() -> None:
    """Fill the sidebar's session slot from the CURRENT transcript."""
    asked = sum(1 for t in st.session_state.get(_TRANSCRIPT_KEY, [])
                if t.get("role") == "user")
    with _session_slot.container():
        if asked:
            st.caption(f"_{asked} question{'s' if asked != 1 else ''} in this "
                       "session — follow-ups keep the whole thread as context._")
        else:
            st.caption("_Ask anything; follow-up questions keep the earlier "
                       "answers as context until you start a new chat._")


_render_session_caption()

# ══════════════════════════════════════════════════════════════════════════
# MAIN PANE
# ══════════════════════════════════════════════════════════════════════════
st.title("🤖 AI Assistant")
st.caption("Ask about the numbers on any tab, the calculations behind them, or "
           "the design decisions recorded in the repo.")

if not _API_KEY:
    st.error("**No Anthropic API key configured.**")
    st.markdown(
        """
The assistant needs a key to reach the Claude API. Set it in **either** place:

* **Streamlit Community Cloud** — *Manage app → Settings → Secrets*:

  ```toml
  ANTHROPIC_API_KEY = "sk-ant-…"
  ```

* **Locally** — either export `ANTHROPIC_API_KEY` in your shell, or create
  `.streamlit/secrets.toml` with the same line. That file is git-ignored, so
  the key is never committed.

Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
Everything else on this page works once the key is present — no other setup.
""")
    st.stop()

st.session_state.setdefault(_HISTORY_KEY, [])
st.session_state.setdefault(_TRANSCRIPT_KEY, [])

_SUGGESTIONS = [
    "What is the target book holding right now, and what changed since the "
    "previous publish?",
    "Strategy Health is red — which sleeves tripped, on which monitor, and "
    "what do the thresholds say that means?",
    "How is each sleeve's target weight actually computed? Walk me through the "
    "calculation and cite the code.",
    "Compare the Balanced profile's backtest to Buy & Hold — return, drawdown "
    "and Sharpe — and say over which window.",
    "Why was VEGN removed from the portfolio?",
]

if not st.session_state[_TRANSCRIPT_KEY]:
    with st.container(border=True):
        st.markdown("**Try one of these**")
        _cols = st.columns(len(_SUGGESTIONS))
        for _i, (_c, _s) in enumerate(zip(_cols, _SUGGESTIONS)):
            with _c:
                if st.button(_s if len(_s) < 60 else _s[:57] + "…",
                             key=f"suggest_{_i}", use_container_width=True,
                             help=_s):
                    st.session_state["assistant_pending"] = _s
                    st.rerun()

# ── replay the visible transcript ───────────────────────────────────────────
def _render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"], avatar="🤖" if turn["role"] == "assistant" else None):
        if turn.get("reasoning"):
            with st.expander("🧠 Reasoning", expanded=False):
                st.markdown(turn["reasoning"])
        for call in turn.get("tools", []):
            icon = "⚠️" if call.get("error") else "🔎"
            with st.expander(f"{icon} `{call['name']}` "
                             f"{_summarise_input(call.get('input', {}))}",
                             expanded=False):
                st.code(str(call.get("result", ""))[:8000], language="text")
        if turn.get("notes"):
            st.info(" · ".join(turn["notes"]), icon="⏳")
        st.markdown(turn["text"])
        if turn.get("usage"):
            st.caption(turn["usage"])


def _summarise_input(payload: dict) -> str:
    """A short, readable rendering of a tool call's arguments for the header."""
    for key in ("path", "pattern", "path_glob"):
        if payload.get(key):
            extra = payload.get("pointer") or payload.get("mode") or ""
            return f"`{payload[key]}`" + (f" → `{extra}`" if extra else "")
    return ""


# ── ask, then the conversation newest-last ──────────────────────────────────
# The router runs every sub-app inside a container, and Streamlit only pins
# ``st.chat_input`` to the viewport bottom when it is a *top-level* element —
# in a container it renders inline, wherever it is called.  So it is called
# BEFORE the transcript: the box keeps a fixed position under the header while
# the conversation grows beneath it, instead of drifting down the page and
# ending up buried between old turns.
_pending = st.session_state.pop("assistant_pending", None)
_typed = st.chat_input("Ask about the book, a backtest number, a calculation, "
                       "or a design decision…")
_question = _pending or _typed

for _turn in st.session_state[_TRANSCRIPT_KEY]:
    _render_turn(_turn)

if _question:
    st.session_state[_TRANSCRIPT_KEY].append({"role": "user", "text": _question})
    _render_turn({"role": "user", "text": _question})

    _history = st.session_state[_HISTORY_KEY] + [{"role": "user", "content": _question}]

    with st.chat_message("assistant", avatar="🤖"):
        _reason_box = st.expander("🧠 Reasoning", expanded=False) if _show_reasoning else None
        _reason_slot = _reason_box.empty() if _reason_box is not None else None
        _tool_slot = st.container()
        _note_slot = st.empty()
        _answer_slot = st.empty()
        _usage_slot = st.empty()

        _answer, _reasoning, _tools, _usage = "", "", [], None
        _notes: list[str] = []
        _failed = None
        try:
            with st.spinner(f"{_spec.label} is reading the repo…"):
                for _ev in ac.stream_answer(
                        api_key=_API_KEY, spec=_spec, messages=_history,
                        system=_system_prompt(_artifact_mtimes()),
                        show_reasoning=_show_reasoning):
                    kind = _ev["type"]
                    if kind == "text":
                        _answer += _ev["text"]
                        _answer_slot.markdown(_answer)
                    elif kind == "thinking":
                        _reasoning += _ev["text"]
                        if _reason_slot is not None:
                            _reason_slot.markdown(_reasoning)
                    elif kind == "tool":
                        _tools.append(_ev)
                        with _tool_slot:
                            icon = "⚠️" if _ev.get("error") else "🔎"
                            with st.expander(
                                    f"{icon} `{_ev['name']}` "
                                    f"{_summarise_input(_ev.get('input', {}))}",
                                    expanded=False):
                                st.code(str(_ev.get("result", ""))[:8000],
                                        language="text")
                    elif kind == "note":
                        # e.g. the tool budget ran out — say so above the
                        # answer rather than letting the turn end silently.
                        _notes.append(_ev["text"])
                        _note_slot.info(" · ".join(_notes), icon="⏳")
                    elif kind == "usage":
                        _usage = _ev
                    elif kind == "done":
                        st.session_state[_HISTORY_KEY] = _ev["messages"]
        except Exception as _exc:                          # noqa: BLE001
            _failed = f"{_exc.__class__.__name__}: {_exc}"

        if _failed:
            _answer_slot.error(
                f"**The request failed.** {_failed}\n\n"
                "If this is an authentication or permission error, check the key "
                "in your Streamlit secrets. If it is a `not_found_error`, this "
                "subscription is not entitled to "
                f"`{_spec.model_id}` — pick another model in the sidebar.")
        else:
            _answer_slot.markdown(_answer or "_(no answer returned)_")

        _usage_line = ""
        if _usage:
            _usage_line = (f"_{_spec.label} · {_usage['input']:,} in / "
                           f"{_usage['output']:,} out tokens · "
                           f"{_usage['cache_read']:,} cached_")
            _usage_slot.caption(_usage_line)

    st.session_state[_TRANSCRIPT_KEY].append({
        "role": "assistant",
        "text": _answer or (f"⚠️ {_failed}" if _failed else "_(no answer returned)_"),
        "reasoning": _reasoning,
        "tools": _tools,
        "notes": _notes,
        "usage": _usage_line,
    })

    _render_session_caption()          # now that this turn is in the transcript

# record this page's own render, like every other app
fr.record_refresh("ASSISTANT", kind="assistant", app_label="🤖 AI Assistant")
