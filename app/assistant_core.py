"""Knowledge + model layer behind the 🤖 AI Assistant page.

The assistant answers questions about *this* app — the numbers the tabs render,
the calculations behind them, and the design decisions recorded in the repo's
markdown.  Accuracy is the whole point, so it never answers from a general
prior about trading: every answer is grounded in two things this module supplies.

1. **A state pack** (``app_state_pack``) — the committed artifacts the pages
   themselves read: the published target book, the executed book, the daily
   audit trail, the strategy-health snapshot, the Overall backtest results and
   each ticker sleeve's backtest/sweep summary.  It is compact (long series are
   pruned) and is embedded in the system prompt on every turn, so the common
   "what does the book hold / is the strategy healthy / what did we execute"
   questions are answered with zero tool round-trips.

2. **Read-only repo tools** (``TOOLS`` / ``execute_tool``) — grep, file read,
   JSON drill-down and dataframe head/tail/describe over the working tree, so a
   deeper question ("*why* is GDX capped at 18%?", "what does the waterfill do?")
   is answered from the actual source and docs rather than from memory.

Both halves are deliberately read-only and path-confined (``_safe_path``): the
assistant can read the repo, never write to it, never reach outside it, and
never open a secrets file.

Model choice
------------
``discover_models`` asks the Messages API which models the configured key can
actually use (``GET /v1/models``) and annotates them from ``MODEL_CATALOG`` —
so the picker shows this subscription's real entitlements rather than a
hardcoded guess.  Capability flags (adaptive thinking, effort levels, output
cap) are per-model, because sending ``output_config.effort`` to a model that
predates it is a 400.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = _APP_DIR.parent


# ════════════════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModelSpec:
    """One selectable model plus the request knobs it actually accepts."""
    model_id: str
    label: str
    blurb: str
    max_output: int = 16_000
    adaptive_thinking: bool = False      # thinking={"type": "adaptive"}
    effort: str | None = None            # output_config={"effort": ...}
    rank: int = 99                       # lower sorts first — capability order
    input_price: float | None = None     # USD per million input tokens
    output_price: float | None = None    # USD per million output tokens

    @property
    def supports_reasoning(self) -> bool:
        return self.adaptive_thinking

    @property
    def priced(self) -> bool:
        return self.input_price is not None and self.output_price is not None

    @property
    def cost_index(self) -> float | None:
        """Relative cost of one question here, lower being cheaper.

        Weighted toward input because this app's turns are: the grounding
        prompt is ~18 k tokens of app state and documentation, while a typical
        answer is a few hundred. Ranking on the sticker output price alone
        would misorder models for *this* workload.
        """
        if not self.priced:
            return None
        return self.input_price + self.output_price / 5.0


#: Curated metadata for the models we know how to drive.  ``discover_models``
#: intersects this with what the key is actually entitled to; anything the API
#: returns that is missing here still shows up, with capabilities inferred from
#: its version (``_infer_spec``).
MODEL_CATALOG: dict[str, ModelSpec] = {
    m.model_id: m for m in (
        ModelSpec("claude-opus-5", "Claude Opus 5",
                  "Strongest multi-step analysis and tool use. Pick this for "
                  "questions that need several look-ups chained together.",
                  max_output=128_000, adaptive_thinking=True, effort="high", rank=0,
                  input_price=5.0, output_price=25.0),
        ModelSpec("claude-fable-5", "Claude Fable 5",
                  "Most capable; slowest and priciest. For the hardest questions.",
                  max_output=128_000, adaptive_thinking=True, effort="high", rank=1,
                  input_price=10.0, output_price=50.0),
        ModelSpec("claude-opus-4-8", "Claude Opus 4.8",
                  "Previous Opus generation; strong reasoning.",
                  max_output=128_000, adaptive_thinking=True, effort="high", rank=2,
                  input_price=5.0, output_price=25.0),
        ModelSpec("claude-opus-4-7", "Claude Opus 4.7",
                  "Older Opus generation.",
                  max_output=128_000, adaptive_thinking=True, effort="high", rank=3,
                  input_price=5.0, output_price=25.0),
        ModelSpec("claude-opus-4-6", "Claude Opus 4.6",
                  "Older Opus generation.",
                  max_output=64_000, adaptive_thinking=True, effort="high", rank=4,
                  input_price=5.0, output_price=25.0),
        ModelSpec("claude-sonnet-5", "Claude Sonnet 5",
                  "A middle option — most of Opus's capability at ~60% of the "
                  "cost. Introductory pricing runs to 2026-08-31.",
                  max_output=128_000, adaptive_thinking=True, effort="high", rank=5,
                  input_price=3.0, output_price=15.0),
        ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6",
                  "Previous Sonnet generation.",
                  max_output=64_000, adaptive_thinking=True, effort="high", rank=6,
                  input_price=3.0, output_price=15.0),
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5",
                  "Cheapest and fastest. Great for look-ups and single-step "
                  "questions; weaker when an answer needs many chained tools.",
                  max_output=64_000, adaptive_thinking=False, effort=None, rank=7,
                  input_price=1.0, output_price=5.0),
    )
}

#: The most *capable* model — labelled as such in the picker, and the fallback
#: default when nothing in the entitled set carries a price.
PREFERRED_MODEL = "claude-opus-5"

_FAMILY_VER = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?")
_LEGACY_VER = re.compile(r"^claude-(\d+)-(\d+)-(opus|sonnet|haiku)")


def _infer_spec(model_id: str, display_name: str | None = None,
                max_output: int | None = None) -> ModelSpec:
    """Best-effort capabilities for a model released after this file was written.

    Adaptive thinking and ``output_config.effort`` arrived with the 4.6
    generation; anything at or above that (or any 5-series) gets them, anything
    older is driven as a plain chat model.  Guessing *down* is the safe
    direction — an unsupported knob is a 400, a missing one is only slower.
    """
    m = _FAMILY_VER.match(model_id)
    if m:
        family, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
        modern = (major, minor) >= (4, 6)
        return ModelSpec(
            model_id, display_name or model_id,
            "Available on this API key.",
            max_output=max_output or (128_000 if modern else 8_192),
            adaptive_thinking=modern,
            effort="high" if modern else None,
            rank=20 + {"fable": 0, "mythos": 0, "opus": 1,
                       "sonnet": 2, "haiku": 3}.get(family, 4),
        )
    if _LEGACY_VER.match(model_id):                       # claude-3-7-sonnet-…
        return ModelSpec(model_id, display_name or model_id,
                         "Legacy model.", max_output=max_output or 8_192, rank=80)
    return ModelSpec(model_id, display_name or model_id,
                     "Available on this API key.",
                     max_output=max_output or 8_192, rank=90)


def resolve_api_key() -> str | None:
    """The Anthropic key, from the environment or Streamlit secrets.

    ``ANTHROPIC_API_KEY`` wins so a local shell can override; on Streamlit
    Community Cloud the key lives in the app's secrets (never in git —
    ``.streamlit/secrets.toml`` is git-ignored).
    """
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env and env.strip():
        return env.strip()
    try:                                                  # secrets may not exist
        import streamlit as st

        for getter in (lambda: st.secrets["ANTHROPIC_API_KEY"],
                       lambda: st.secrets["anthropic"]["api_key"]):
            try:
                val = getter()
            except Exception:
                continue
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    return None


def build_client(api_key: str):
    """An Anthropic client.  Import is local so the rest of the app — and the
    test suite — runs fine without the SDK installed."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=180.0)


def discover_models(api_key: str) -> tuple[list[ModelSpec], str | None]:
    """(models this key can use, error).  Falls back to the catalogue.

    The Models API is the only honest answer to "which models does my
    subscription have" — entitlements differ per organisation, so a hardcoded
    list would eventually offer a model the key gets a 404 on.  If the call
    fails (offline, bad key) the catalogue is returned so the picker still
    renders and the real error surfaces on the first message instead.
    """
    try:
        client = build_client(api_key)
        listed = client.models.list(limit=100).data
    except Exception as exc:                              # noqa: BLE001 — surfaced to UI
        return sorted(MODEL_CATALOG.values(), key=lambda s: s.rank), _short_error(exc)

    specs: list[ModelSpec] = []
    for m in listed:
        mid = getattr(m, "id", None)
        if not mid:
            continue
        known = MODEL_CATALOG.get(mid)
        if known is not None:
            specs.append(known)
        else:
            specs.append(_infer_spec(mid, getattr(m, "display_name", None),
                                     getattr(m, "max_tokens", None)))
    if not specs:
        return sorted(MODEL_CATALOG.values(), key=lambda s: s.rank), None
    return sorted(specs, key=lambda s: (s.rank, s.model_id)), None


def spec_for(model_id: str) -> ModelSpec:
    """The capability record for a model id — catalogued or inferred."""
    return MODEL_CATALOG.get(model_id) or _infer_spec(model_id)


def cheapest_model(specs: list[ModelSpec]) -> str | None:
    """The lowest ``cost_index`` among the entitled models that carry a price.

    A model discovered from the API but absent from ``MODEL_CATALOG`` has no
    price, and is therefore never named "most economical" — we would be
    guessing.  Ties break toward the more capable model (lower ``rank``).
    """
    priced = [s for s in specs if s.priced]
    if not priced:
        return None
    return min(priced, key=lambda s: (s.cost_index, s.rank)).model_id


def most_capable_model(specs: list[ModelSpec]) -> str | None:
    """The best-ranked entitled model, for labelling the other end of the scale."""
    return min(specs, key=lambda s: s.rank).model_id if specs else None


def default_model(specs: list[ModelSpec]) -> str:
    """The pre-selected model — the cheapest one this subscription can use.

    Cost is the default because most questions here are look-ups the state pack
    already answers, and the picker is one click away when a question needs
    more.  It is *computed* from the catalogue's per-MTok prices rather than
    hardcoded, so the day a cheaper model ships it wins automatically.  With no
    priced model in the entitled set we fall back to the most capable one
    rather than picking arbitrarily.
    """
    ids = [s.model_id for s in specs]
    if not ids:
        return PREFERRED_MODEL
    return (cheapest_model(specs) or most_capable_model(specs)
            or (PREFERRED_MODEL if PREFERRED_MODEL in ids else ids[0]))


def _short_error(exc: Exception) -> str:
    msg = getattr(exc, "message", None) or str(exc)
    return msg.strip().splitlines()[0][:300] if msg else exc.__class__.__name__


# ════════════════════════════════════════════════════════════════════════════
# REPO ACCESS  (read-only, path-confined)
# ════════════════════════════════════════════════════════════════════════════
#: Only text the assistant can usefully quote.  Binary model/artifact blobs are
#: excluded — they'd be unreadable and huge.
_TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml",
                  ".cfg", ".ini", ".html", ".sh", ".ps1", ".bat", ".sql", ".example"}
#: Directories that are noise or not ours.
_DENY_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints",
              ".pytest_cache", ".mypy_cache"}
#: Nothing that could hold a credential is readable, whatever else says.  These
#: are git-ignored anyway, but the guard has to hold for a local checkout too.
_DENY_NAME_GLOBS = ("*.env", ".env*", "*secret*", "*credential*", "*.key", "*.pem",
                    "*.pfx", "*token*", "id_rsa*")

_MAX_READ_CHARS = 60_000


class RepoAccessError(Exception):
    """A tool asked for something outside the read-only repo sandbox."""


def _safe_path(rel: str, *, require_text: bool = True) -> Path:
    """Resolve ``rel`` under the repo root or refuse.

    ``resolve()`` collapses ``..`` and follows symlinks *before* the
    containment check, so neither can be used to climb out of the tree.
    """
    if not rel or not str(rel).strip():
        raise RepoAccessError("empty path")
    candidate = (REPO_ROOT / str(rel).strip().lstrip("/")).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        raise RepoAccessError(f"path escapes the repository: {rel}") from None
    parts = candidate.relative_to(REPO_ROOT).parts
    if any(p in _DENY_DIRS for p in parts):
        raise RepoAccessError(f"path is not readable: {rel}")
    lowered = candidate.name.lower()
    if any(fnmatch.fnmatch(lowered, g) for g in _DENY_NAME_GLOBS):
        raise RepoAccessError(f"path is not readable: {rel}")
    if require_text and candidate.suffix.lower() not in _TEXT_SUFFIXES:
        raise RepoAccessError(
            f"unsupported file type '{candidate.suffix}' — readable types: "
            + ", ".join(sorted(_TEXT_SUFFIXES)))
    return candidate


def _iter_repo_files(pattern: str, *, limit: int = 400) -> list[Path]:
    """Every readable repo file matching a glob, deterministically ordered."""
    out: list[Path] = []
    for p in sorted(REPO_ROOT.glob(pattern)):
        if not p.is_file():
            continue
        try:
            _safe_path(str(p.relative_to(REPO_ROOT)))
        except (RepoAccessError, ValueError):
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# ── tool implementations ────────────────────────────────────────────────────
def tool_list_files(path_glob: str = "*.md") -> str:
    files = _iter_repo_files(path_glob)
    if not files:
        return f"No readable files match {path_glob!r}."
    rows = [f"{rel(p):<58} {p.stat().st_size:>9,} bytes" for p in files]
    return f"{len(files)} file(s) matching {path_glob!r}:\n" + "\n".join(rows)


def tool_read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"{path} does not exist."
    start = max(1, int(start_line))
    span = max(1, min(int(max_lines), 2_000))
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[start - 1: start - 1 + span]
    body = "\n".join(f"{start + i:>6}  {ln}" for i, ln in enumerate(chunk))
    if len(body) > _MAX_READ_CHARS:
        body = body[:_MAX_READ_CHARS] + "\n… [truncated — narrow the line range]"
    tail = ""
    if start - 1 + span < len(lines):
        tail = (f"\n… {len(lines) - (start - 1 + span)} more line(s); "
                f"re-read with start_line={start + span}")
    return f"{rel(p)} (lines {start}–{start + len(chunk) - 1} of {len(lines)}):\n{body}{tail}"


#: The default search scope.  It deliberately spans the markdown record AND the
#: source: how a thing WORKS lives in ``app/*.py`` and its docstrings far more
#: often than in a .md file — the 12:00-UTC bar convention, for one, is
#: documented only in ``app/freshness.py``'s module docstring.  Searching docs
#: alone sent the assistant hunting through 17 k lines of app source by hand.
_DEFAULT_SEARCH_GLOBS = "*.md,docs/*.md,app/*.py,scripts/*.py,*.py"


def tool_search_repo(pattern: str, path_glob: str = _DEFAULT_SEARCH_GLOBS,
                     max_results: int = 40, ignore_case: bool = True,
                     context: int = 0) -> str:
    """Regex search over one or more comma-separated globs, with context lines."""
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return f"Invalid regular expression {pattern!r}: {exc}"
    cap = max(1, min(int(max_results), 200))
    ctx = max(0, min(int(context), 10))
    globs = [g.strip() for g in str(path_glob).split(",") if g.strip()]

    seen: set[Path] = set()
    hits: list[str] = []
    capped = False
    for glob in globs:
        for p in _iter_repo_files(glob):
            if p in seen:
                continue
            seen.add(p)
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                if not rx.search(line):
                    continue
                if ctx:
                    lo, hi = max(0, n - 1 - ctx), min(len(lines), n + ctx)
                    block = "\n".join(
                        f"{'>' if i + 1 == n else ' '} {i + 1:>5}  {lines[i][:240]}"
                        for i in range(lo, hi))
                    hits.append(f"{rel(p)}:{n}:\n{block}")
                else:
                    hits.append(f"{rel(p)}:{n}: {line.strip()[:240]}")
                if len(hits) >= cap:
                    capped = True
                    break
            if capped:
                break
        if capped:
            break

    if not hits:
        return (f"No match for {pattern!r} in {path_glob!r}. "
                "Widen the glob (e.g. '**/*.py') or loosen the pattern.")
    head = f"{len(hits)} match(es) for {pattern!r}{' (capped)' if capped else ''}:"
    body = "\n".join(hits)
    if len(body) > _MAX_READ_CHARS:
        body = body[:_MAX_READ_CHARS] + "\n… [truncated — narrow the pattern]"
    return f"{head}\n{body}"


def _drill(obj: Any, pointer: str) -> Any:
    """Walk ``a.b.0.c`` through nested dicts/lists."""
    cur = obj
    for token in [t for t in pointer.replace("/", ".").split(".") if t]:
        if isinstance(cur, list):
            try:
                cur = cur[int(token)]
            except (ValueError, IndexError):
                raise KeyError(f"no element {token!r} in a list of {len(cur)}") from None
        elif isinstance(cur, dict):
            if token not in cur:
                raise KeyError(f"no key {token!r}; available: "
                               f"{', '.join(list(cur)[:25])}")
            cur = cur[token]
        else:
            raise KeyError(f"cannot index into {type(cur).__name__} with {token!r}")
    return cur


def tool_read_json(path: str, pointer: str = "", max_chars: int = 12_000) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"{path} does not exist."
    if p.suffix.lower() != ".json":
        return f"{path} is not a .json file — use read_file or read_table."
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Could not parse {path}: {exc}"
    if pointer:
        try:
            data = _drill(data, pointer)
        except KeyError as exc:
            return f"{path}: {exc}"
    pruned = prune(data)
    text = json.dumps(pruned, indent=1, ensure_ascii=False, default=str)
    cap = max(500, min(int(max_chars), 40_000))
    if len(text) > cap:
        text = text[:cap] + "\n… [truncated — pass a deeper `pointer` to narrow]"
    return f"{rel(p)}{(' → ' + pointer) if pointer else ''}:\n{text}"


def tool_read_table(path: str, mode: str = "head", n: int = 15,
                    columns: list[str] | None = None) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"{path} does not exist."
    if p.suffix.lower() != ".csv":
        return f"{path} is not a .csv file — use read_file or read_json."
    import pandas as pd

    try:
        df = pd.read_csv(p)
    except Exception as exc:                              # noqa: BLE001
        return f"Could not read {path}: {exc}"
    shape = f"{rel(p)} — {len(df):,} rows × {len(df.columns)} columns"
    if columns:
        keep = [c for c in columns if c in df.columns]
        if not keep:
            return f"{shape}\nNone of {columns} exist. Columns: {list(df.columns)}"
        df = df[keep]
    rows = max(1, min(int(n), 200))
    if mode == "shape":
        return f"{shape}\ncolumns: {list(df.columns)}"
    if mode == "columns":
        return f"{shape}\n" + "\n".join(f"  {c}  ({df[c].dtype})" for c in df.columns)
    if mode == "describe":
        return f"{shape}\n{df.describe(include='all').to_string()[:_MAX_READ_CHARS]}"
    frame = df.tail(rows) if mode == "tail" else df.head(rows)
    return f"{shape} — {mode}({rows}):\n{frame.to_string()[:_MAX_READ_CHARS]}"


# ── live signals: run the real engines, don't read their source ─────────────
#: Which engine owns each sleeve.  Questions about "is X about to trigger" are
#: answered by RUNNING the engine, never by reverse-engineering it out of the
#: source — which is what an assistant without this tool is forced to attempt,
#: and why such questions used to burn every tool round without landing.
_CT_KEYS = {"BTC", "MSTR", "MSTU", "ETH"}
_GLDM_KEYS = {"GLDM": "trend", "UGL": "trend", "GDX": "miners", "NUGT": "miners",
              "GDXM": "miners"}

#: Engine runs are expensive (the CT model load alone is ~20 s) and the answer
#: only changes when a bar closes, so results are memoised for the process.
_SIGNAL_CACHE: dict[str, Any] = {}


def _engine_import(name: str):
    import importlib

    return importlib.import_module(name)


def _ct_results() -> list[dict]:
    if "ct" not in _SIGNAL_CACHE:
        _SIGNAL_CACHE["ct"] = _engine_import("btc_ct_engine").run_btc_ct()
    return _SIGNAL_CACHE["ct"]


def _summarise_result(r: dict) -> dict:
    """The decision-relevant fields of an engine result, without the curves."""
    dec = r.get("decision") or {}
    pos = r.get("pos") or {}
    return {
        "key": r.get("key"),
        "name": r.get("name"),
        "as_of_bar_start": str(getattr(r.get("as_of"), "date", lambda: r.get("as_of"))()),
        "decision": dec.get("label"),
        "state": dec.get("state"),
        "exits_next_bar": bool(dec.get("exits_next_bar", False)),
        "alert": r.get("alert"),
        "bull_regime": r.get("bull_regime"),
        "in_position": pos.get("in_pos"),
        "entry_px": pos.get("entry_px"),
        "unrealised_pct": pos.get("upnl"),
        "stop_px": pos.get("stop_px"),
        "last_close": r.get("last_close"),
        "day_change_pct": r.get("dchg"),
        "ma_value": r.get("ma_val"),
        "n_trades": r.get("n_trades"),
        "win_rate_pct": r.get("win_rate"),
    }


def tool_live_signal(sleeve: str, diagnostics: bool = True,
                     lookback_bars: int = 8) -> str:
    """Current signal for one sleeve, from the engine the app itself runs."""
    key = str(sleeve).strip().upper()
    if key in _CT_KEYS:
        rows = [_summarise_result(r) for r in _ct_results() if r.get("key") == key]
        if not rows:
            return f"The CT engine returned no result for {key}."
        out: dict[str, Any] = {"sleeve": key, "signal": rows[0]}
        if diagnostics:
            diag = _SIGNAL_CACHE.get("ct_diag")
            if diag is None:
                diag = _engine_import("btc_ct_engine").live_diagnostics(
                    lookback=max(2, min(int(lookback_bars), 30)))
                _SIGNAL_CACHE["ct_diag"] = diag
            out["diagnostics"] = diag
        return json.dumps(prune(out, max_list=30), indent=1, default=str)

    if key in _GLDM_KEYS:
        eng = _engine_import("gldm_engine")
        mode = _GLDM_KEYS[key]
        cache_key = f"gldm_{mode}"
        if cache_key not in _SIGNAL_CACHE:
            _SIGNAL_CACHE[cache_key] = (eng.run_gldm_miners() if mode == "miners"
                                        else eng.run_gldm_trend())
        rows = [_summarise_result(r) for r in _SIGNAL_CACHE[cache_key]]
        return json.dumps({"sleeve": key, "signals": rows}, indent=1, default=str)

    oc = _engine_import("overall_core")
    try:
        cfg = oc.overall_config(key)
    except Exception:
        known = sorted(_CT_KEYS | set(_GLDM_KEYS)) + list(
            _engine_import("ticker_config").APP_KEYS)
        return f"Unknown sleeve {key!r}. Known sleeves: {', '.join(known)}."
    cache_key = f"asset_{key}"
    if cache_key not in _SIGNAL_CACHE:
        _SIGNAL_CACHE[cache_key] = oc.run_asset(cfg)
    rows = [_summarise_result(r) for r in _SIGNAL_CACHE[cache_key]]
    return json.dumps({"sleeve": key, "signals": rows}, indent=1, default=str)


#: name → (callable, JSON schema).  The schemas are the model's whole view of
#: what it may do, so they spell out the *purpose* of each argument.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "live_signal",
        "description": (
            "THE current signal for one sleeve, computed by running the app's "
            "OWN engine — not by reading its source. Use this for ANY question "
            "about what a sleeve is doing now or what would make it trigger: "
            "'is BTC about to fire a buy?', 'why is SOXX flat?', 'how close is "
            "GDX to an exit?'. Returns the decision, regime, position and — for "
            "BTC/MSTR/MSTU/ETH — full gate diagnostics: per-bar predicted vs "
            "realised high/low, the U1 components against their thresholds, and "
            "exactly what the NEXT bar must do to trigger an entry (including "
            "whether it is arithmetically impossible). Prefer this over reading "
            "engine source; one call beats ten greps."),
        "input_schema": {
            "type": "object",
            "properties": {
                "sleeve": {"type": "string",
                           "description": "BTC, MSTR, MSTU, ETH, GLDM, UGL, GDX, "
                                          "NUGT, SOXX, GRID, XLE, REMX, WGMI, "
                                          "PBW or ARTY."},
                "diagnostics": {"type": "boolean",
                                "description": "Include the gate breakdown and "
                                               "recent bars (default true)."},
                "lookback_bars": {"type": "integer",
                                  "description": "Bars of predicted-vs-actual "
                                                 "history (default 8)."},
            },
            "required": ["sleeve"],
        },
    },
    {
        "name": "search_repo",
        "description": (
            "Regex-search the repository's text files, with surrounding context. "
            "Use this to locate where something is documented or implemented — "
            "e.g. 'waterfill|18%' for a design decision, or 'def priority' for "
            "the calculation. The default scope covers the markdown record AND "
            "the app/script source, because HOW something works is usually in "
            "the code and its docstrings, not only in the .md files. Returns "
            "path:line: text, so one search often removes the need to read_file "
            "at all."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path_glob": {"type": "string",
                              "description": "One glob, or several comma-separated, "
                                             "relative to the repo root — e.g. "
                                             "'app/*.py', '*.md,app/*.py', "
                                             "'data/**/*.json'. Defaults to the "
                                             "docs plus app/ and scripts/ source."},
                "max_results": {"type": "integer", "description": "Cap (default 40)."},
                "context": {"type": "integer",
                            "description": "Lines of context around each hit "
                                           "(default 0, max 10). Use 2-4 to read "
                                           "a definition without a second call."},
                "ignore_case": {"type": "boolean", "description": "Default true."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file from the repository (markdown docs, Python source, "
            "config). Use after search_repo to read the surrounding context, or "
            "directly when you know the path. Line-numbered; page with start_line."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Repo-relative path, e.g. 'OVERALL_STRATEGY.md' "
                                        "or 'app/overall_core.py'."},
                "start_line": {"type": "integer", "description": "1-based. Default 1."},
                "max_lines": {"type": "integer", "description": "Default 400, max 2000."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List repository files matching a glob, with sizes. Use to "
                       "discover what exists before reading.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path_glob": {"type": "string",
                              "description": "e.g. '*.md', 'data/overall/*.json', "
                                             "'data/*/backtest_results.json'."},
            },
            "required": ["path_glob"],
        },
    },
    {
        "name": "read_json",
        "description": (
            "Read a JSON artifact, optionally drilling to a sub-object. The state "
            "pack in your context is PRUNED — use this to get a full value the "
            "pack elided (long series, every sleeve field, an archived book). "
            "Pointer syntax: 'sleeves.0.dd' or 'portfolio/profiles'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "e.g. 'data/overall/strategy_health.json'."},
                "pointer": {"type": "string",
                            "description": "Dotted/slashed path into the document. "
                                           "Optional."},
                "max_chars": {"type": "integer", "description": "Default 12000."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_table",
        "description": (
            "Inspect a CSV — price history, macro features, health history. Modes: "
            "'head', 'tail', 'describe', 'shape', 'columns'. Use 'shape'/'columns' "
            "first on an unfamiliar file, then 'tail' for the most recent rows."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "e.g. 'data/overall/health_history.csv'."},
                "mode": {"type": "string",
                         "enum": ["head", "tail", "describe", "shape", "columns"]},
                "n": {"type": "integer", "description": "Rows for head/tail (default 15)."},
                "columns": {"type": "array", "items": {"type": "string"},
                            "description": "Restrict to these columns."},
            },
            "required": ["path"],
        },
    },
]

_DISPATCH: dict[str, Callable[..., str]] = {
    "live_signal": tool_live_signal,
    "search_repo": tool_search_repo,
    "read_file": tool_read_file,
    "list_files": tool_list_files,
    "read_json": tool_read_json,
    "read_table": tool_read_table,
}


def execute_tool(name: str, payload: dict[str, Any]) -> tuple[str, bool]:
    """Run one tool call.  Returns ``(text, is_error)`` — never raises, because
    a raised exception would abort the whole answer where a returned error lets
    the model correct itself and try again."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool {name!r}. Available: {', '.join(_DISPATCH)}.", True
    try:
        kwargs = {k: v for k, v in (payload or {}).items() if v is not None}
        return str(fn(**kwargs)), False
    except RepoAccessError as exc:
        return f"Refused: {exc}", True
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}", True
    except Exception as exc:                              # noqa: BLE001
        return f"{name} failed: {exc.__class__.__name__}: {exc}", True


# ════════════════════════════════════════════════════════════════════════════
# STATE PACK — what the tabs are showing right now
# ════════════════════════════════════════════════════════════════════════════
#: Keys whose values are per-day equity/price series: informative to plot,
#: pure noise in a prompt.  Dropped wholesale; ``read_json`` can fetch them.
_SERIES_KEYS = {"series", "equity", "curve", "curves", "dates", "history",
                "daily", "trades_log", "log"}


def prune(obj: Any, *, max_list: int = 12, _depth: int = 0) -> Any:
    """Shrink an artifact to its decision-relevant shape.

    Long lists become a head/tail sample with the true length recorded, so the
    model can see the data's scale without being handed 4 000 dates — and knows
    to call ``read_json`` when it needs an elided value.
    """
    if _depth > 12:
        return "…"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SERIES_KEYS and isinstance(v, (list, dict)):
                n = len(v)
                out[str(k)] = f"<{n} points elided — read_json for the full series>"
                continue
            out[str(k)] = prune(v, max_list=max_list, _depth=_depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > max_list:
            head = [prune(x, max_list=max_list, _depth=_depth + 1) for x in obj[:max_list]]
            return head + [f"<{len(obj) - max_list} more of {len(obj)} elided>"]
        return [prune(x, max_list=max_list, _depth=_depth + 1) for x in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"__unavailable__": f"{path.name}: {exc.__class__.__name__}"}


def app_state_pack(max_list: int = 12) -> dict[str, Any]:
    """The committed artifacts every page reads, pruned for the prompt.

    Deliberately free of wall-clock timestamps: the pack is embedded in the
    cached system prompt, and a ``now()`` in there would invalidate the prompt
    cache on every single turn.  Freshness is expressed by the artifacts' own
    ``as_of`` / ``generated_at_utc`` fields, which is what the pages show too.
    """
    overall = REPO_ROOT / "data" / "overall"
    pack: dict[str, Any] = {}

    for name, fname in (("target_book", "target_book.json"),
                        ("target_book_live", "target_book_live.json"),
                        ("executed_book", "executed_book.json"),
                        ("daily_audit", "daily_audit.json"),
                        ("book_versions", "book_versions.json"),
                        ("overall_results", "overall_results.json"),
                        ("strategy_health", "strategy_health.json")):
        p = overall / fname
        pack[name] = (prune(_load_json(p), max_list=max_list) if p.exists()
                      else {"__missing__": f"data/overall/{fname}"})

    sleeves: dict[str, Any] = {}
    for bt in sorted(REPO_ROOT.glob("data/*/backtest_results.json")):
        sleeves[bt.parent.name] = prune(_load_json(bt), max_list=max_list)
    pack["sleeve_backtests"] = sleeves
    return pack


def _fmt_bytes(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


def doc_index() -> str:
    """A one-line-per-document map of the repo's written record.

    The first ``#`` heading is used as the summary, so the model can pick the
    right document to open without reading all ~350 KB of markdown.
    """
    rows: list[str] = []
    for p in _iter_repo_files("*.md") + _iter_repo_files("docs/*.md"):
        title = ""
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        except OSError:
            pass
        rows.append(f"  {rel(p):<40} {_fmt_bytes(p.stat().st_size):>7}  {title[:90]}")
    return "\n".join(rows)


# ── the app map: what each radio option renders, and where it comes from ────
APP_MAP = """\
The sidebar "Application" radio switches between these pages (streamlit_app.py
routes them; each sub-app also renders the same radio):

  🧭 OVERALL    app/overall_app.py + app/overall_core.py — the cross-asset cockpit.
                Tabs: "🔴 Live — Decision Cockpit", "🕰️ Historical View",
                "📊 Combined Backtesting", "🧠 Strategy & Methodology".
                Live shows per-sleeve decisions, priorities and the blended
                target weights; Combined Backtesting shows the profile table
                (total return / CAGR / MDD / Sharpe / vol) vs Buy & Hold —
                the same numbers as data/overall/overall_results.json.
  ₿  BTC        app/btc_hourly_app.py (+ app/btc_ct_engine.py, app/leading_indicators.py).
                Tabs: Live (rolling now+1h), Historical replay, BTC/MSTR/MSTU/ETH
                Backtesting, MSTR & MSTU Options, Explainability, Leading
                Indicators, Retrain Status.
  🥇 GLDM       app/gldm_hourly_app.py + app/gldm_core.py — gold trend (GLDM·UGL).
  ⛏️  GDXM       the same GLDM app in miners mode (GDX·NUGT).
  SOXX GRID XLE REMX WGMI PBW ARTY
                app/ticker_app.py + app/ticker_core.py, configured by
                app/ticker_config.py. Tabs: Live, Historical replay, per-asset
                Backtesting, Explain.
  🕵️  DAILYAUDIT app/daily_audit_app.py — the freshness trail: per-app signal
                closes, the Overall update and the book publish
                (data/overall/daily_audit.json, app/freshness.py).
  🩺 HEALTH     app/health_app.py + app/health_core.py — the decay monitor.
                M1 book-vs-replay tracking, M2 drawdown tripwires, M3 rolling
                6-month edge vs Buy & Hold, M4 last-20-trade expectancy
                (data/overall/strategy_health.json, thresholds in STRATEGY_HEALTH.md).
  📋 TARGETBOOK app/target_book_app.py — the published, signed IBKR target book
                (data/overall/target_book.json / target_book_live.json).
  ✅ EXECUTEDBOOK app/executed_book_app.py — trades executed and the resulting
                IBKR positions (data/overall/executed_book.json).
  🤖 ASSISTANT  this page (app/assistant_app.py + app/assistant_core.py).

Live tabs RECOMPUTE from market data when opened; the JSON artifacts are the
nightly committed snapshot. When they disagree, the artifact's `as_of` is the
truth about what was *published*, and the Live tab is the truth about *now* —
and `live_signal` is how YOU reach that same live computation."""


BAR_CLOCK = """\
Bar timing — read this before answering any "will it trigger / has it closed"
question, because the two clocks below are the usual source of a wrong answer
--------------------------------------------------------------------------
* BTC · MSTR · MSTU · ETH run on **12:00-UTC-anchored daily bars**. Bar *D*
  covers `[D 12:00Z, D+1 12:00Z)` and its signal becomes knowable the INSTANT
  the bar closes at `D+1 12:00Z` — which is **7:00 AM US Central in summer
  (CDT), 6:00 AM in winter (CST)**.
* Every engine reports `as_of` as the bar **START** date *D*. So the newest
  completed bar is normally *yesterday's* date, and *today's* date is the bar
  still in progress, closing tomorrow morning at 12:00Z.
  A question about "the bar closing tomorrow at 7 AM CT" is therefore about the
  bar whose `as_of` START date is TODAY — one bar AFTER the newest completed
  bar `live_signal` reports.
* Equity sleeves (GLDM, GDX, SOXX, GRID, XLE, REMX, WGMI, PBW, ARTY and their
  leveraged siblings) run on **exchange sessions**: session *D* closes at
  `D 16:00 ET`.
* The daily publish anchors at **7:15 AM US Central** — after the Bitcoin bar
  close, so the book sees BTC's fresh signal and every equity's prior close.
* The full statement of all of this is `app/freshness.py`'s module docstring.

A signal for a bar that has NOT closed is not knowable — the gates read that
bar's realised high/low. What IS knowable, and what you should give, is what
the next bar would have to do: `live_signal` returns exactly that under
`next_bar_entry_requirement`, including whether an entry is arithmetically
impossible at the next close."""


_ANSWERING_RULES = """\
How to answer
-------------
* Ground every number. If it is in the state pack below, use that value and say
  which artifact it came from. If it is not, call a tool and read it. Never
  estimate, interpolate or recall a number from training.
* Prefer the state pack over tools when it already answers the question — it is
  the same data the pages render.
* For "why" questions (why this weight, why this stop, why this ticker was
  dropped), the answer is in the markdown record: search it, read the relevant
  section, and cite the file. The *_EVAL.md files hold the experiments that
  settled each decision; OVERALL_STRATEGY.md, TRADING_STRATEGY.md and
  STRATEGY_HEALTH.md hold the methodology.
* For anything about a CURRENT or IMMINENT signal — "is X about to buy", "will
  the next bar trigger", "why is X flat", "how close is the exit" — call
  `live_signal` FIRST. It runs the app's real engine and returns the gate
  components against their thresholds. Do NOT try to reconstruct a signal by
  reading engine source: that is slow, and the answer it produces is a guess
  about code rather than a reading of data.
* Pick the tool that answers the question in one call. You have a limited tool
  budget per question; if you are three calls in and no closer, stop searching
  and answer with what you have, stating plainly what you could not determine.
  A useful partial answer beats an exhausted budget and no answer at all.
* Cite as `path` or `path:line`. Quote the sentence that decides the point when
  a design decision is contested.
* State the as-of date whenever you quote a book, a health flag or a position:
  this data is a committed nightly snapshot, not a live quote.
* If the artifacts genuinely do not contain the answer — a live intraday price,
  a backtest nobody has run — say so plainly and say what would produce it.
  Never fill a gap with a plausible-sounding number.
* Show the arithmetic when you compute something (a weight sum, a return, a
  drawdown), so the user can check it.
* Be concise and concrete. Markdown tables for anything multi-row.
* This is a personal research/trading tool. Explain what the model and the book
  say; do not add generic investment-advice disclaimers to every answer."""


def build_system_prompt(state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The cached system prompt: identity, app map, doc index, live state pack.

    Returned as content blocks with a cache breakpoint at the end — the whole
    thing is ~4× the size of a typical question and byte-stable between turns,
    so caching it turns a large recurring input cost into a ~10% one.
    """
    pack = app_state_pack() if state is None else state
    body = f"""\
You are the AI Assistant built into a personal quantitative trading app
(the "btc-range-model" repository). You answer questions about THIS app: the
numbers its tabs display, the calculations that produce them, the data behind
them, and the design decisions recorded in its documentation.

{APP_MAP}

{BAR_CLOCK}

{_ANSWERING_RULES}

Documentation index (use read_file / search_repo to open any of these)
---------------------------------------------------------------------
{doc_index()}

Current committed state (data/overall/*.json and data/*/backtest_results.json,
pruned — long series are elided and retrievable with read_json)
---------------------------------------------------------------------
{json.dumps(pack, indent=1, ensure_ascii=False, default=str)}
"""
    return [{"type": "text", "text": body, "cache_control": {"type": "ephemeral"}}]


# ════════════════════════════════════════════════════════════════════════════
# THE CHAT TURN
# ════════════════════════════════════════════════════════════════════════════
#: Tool rounds before the loop stops fetching.  Raised from 8: with
#: ``live_signal`` most questions now resolve in one or two, but a genuinely
#: multi-part question ("compare these three sleeves and say why") needs room,
#: and running out used to mean the user got NOTHING back.
MAX_TOOL_ROUNDS = 12
#: Chat answers, not documents — a cap this size never truncates a real answer
#: and keeps a runaway generation from burning the budget.
ANSWER_MAX_TOKENS = 16_000


def request_kwargs(spec: ModelSpec, *, show_reasoning: bool) -> dict[str, Any]:
    """Per-model request knobs.

    Every one of these is model-gated: ``output_config.effort`` and adaptive
    thinking are rejected by models older than the 4.6 generation. Thinking is
    never explicitly *disabled* — hiding the reasoning is a display choice
    (``display``), and disabling it on the Opus 5 family degrades tool use.
    """
    kwargs: dict[str, Any] = {"max_tokens": min(spec.max_output, ANSWER_MAX_TOKENS)}
    if spec.adaptive_thinking:
        thinking: dict[str, Any] = {"type": "adaptive"}
        if show_reasoning:
            thinking["display"] = "summarized"
        kwargs["thinking"] = thinking
    if spec.effort:
        kwargs["output_config"] = {"effort": spec.effort}
    return kwargs


def stream_answer(*, api_key: str, spec: ModelSpec,
                  messages: list[dict[str, Any]],
                  system: list[dict[str, Any]] | None = None,
                  show_reasoning: bool = False) -> Iterator[dict[str, Any]]:
    """Run one question to completion, yielding UI events as they happen.

    Events: ``{"type": "thinking"|"text", "text": …}`` for streamed output,
    ``{"type": "tool", "name": …, "input": …, "result": …, "error": bool}``
    once a tool has run, ``{"type": "usage", …}`` per API turn, and finally
    ``{"type": "done", "messages": [...]}`` carrying the full transcript
    (assistant content blocks included) to be stored for the next question.

    ``messages`` is mutated only through a local copy; the caller gets the
    updated history in the ``done`` event.
    """
    client = build_client(api_key)
    convo = list(messages)
    sys_blocks = build_system_prompt() if system is None else system
    base = request_kwargs(spec, show_reasoning=show_reasoning)

    for _ in range(MAX_TOOL_ROUNDS):
        with client.messages.stream(
            model=spec.model_id,
            system=sys_blocks,
            tools=TOOLS,
            messages=convo,
            # Caches the growing conversation tail on top of the system
            # breakpoint, so each follow-up question re-reads the prefix
            # instead of re-paying for it.
            cache_control={"type": "ephemeral"},
            **base,
        ) as stream:
            for event in stream:
                if event.type == "text":                  # SDK helper: text delta
                    yield {"type": "text", "text": event.text}
                elif (event.type == "content_block_delta"
                      and getattr(event.delta, "type", "") == "thinking_delta"):
                    yield {"type": "thinking",
                           "text": getattr(event.delta, "thinking", "")}
            final = stream.get_final_message()

        usage = getattr(final, "usage", None)
        if usage is not None:
            yield {"type": "usage",
                   "input": getattr(usage, "input_tokens", 0),
                   "output": getattr(usage, "output_tokens", 0),
                   "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                   "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0}

        convo.append({"role": "assistant", "content": final.content})

        if final.stop_reason == "refusal":
            yield {"type": "text",
                   "text": "\n\n_The model declined to answer this request._"}
            break
        if final.stop_reason != "tool_use":
            break

        results: list[dict[str, Any]] = []
        for block in final.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            text, is_error = execute_tool(block.name, dict(block.input or {}))
            yield {"type": "tool", "name": block.name, "input": dict(block.input or {}),
                   "result": text, "error": is_error}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": text, "is_error": is_error})
        convo.append({"role": "user", "content": results})
    else:
        # Budget exhausted.  Previously this ended the turn with whatever the
        # last round happened to have streamed — for a question the model was
        # still investigating, that is nothing at all, and the user saw a
        # spinner resolve into silence.  Instead, force one final pass with
        # tool use forbidden: the model has to answer from what it already
        # gathered and say what it could not determine.  A partial answer with
        # its gaps named is always more useful than an empty one.
        convo.append({
            "role": "user",
            "content": (
                f"You have used all {MAX_TOOL_ROUNDS} tool calls for this "
                "question. Do not request another. Answer now from what you "
                "have gathered: give your best supported conclusion, show the "
                "figures it rests on, and state plainly which part you could "
                "not determine and what would settle it."),
        })
        yield {"type": "note",
               "text": f"Tool budget ({MAX_TOOL_ROUNDS} rounds) reached — "
                       "answering from what was gathered."}
        try:
            with client.messages.stream(
                model=spec.model_id, system=sys_blocks, tools=TOOLS,
                # Keep the tool definitions (the history contains tool_use
                # blocks that reference them) but forbid another call.
                tool_choice={"type": "none"},
                messages=convo, cache_control={"type": "ephemeral"}, **base,
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        yield {"type": "text", "text": event.text}
                    elif (event.type == "content_block_delta"
                          and getattr(event.delta, "type", "") == "thinking_delta"):
                        yield {"type": "thinking",
                               "text": getattr(event.delta, "thinking", "")}
                closing = stream.get_final_message()
            convo.append({"role": "assistant", "content": closing.content})
        except Exception as exc:                          # noqa: BLE001
            yield {"type": "text",
                   "text": ("\n\n_Ran out of tool rounds, and the closing "
                            f"summary also failed: {exc.__class__.__name__}._")}

    yield {"type": "done", "messages": convo}
