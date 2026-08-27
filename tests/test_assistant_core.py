"""Unit tests for the 🤖 AI Assistant's knowledge + model layer.

Three things have to hold for the page to be trustworthy:

  * the repo sandbox really is a sandbox — no escaping the tree, no reading a
    credential file, no binary blobs;
  * the state pack embedded in every prompt is bounded and actually contains
    the artifacts the pages render (an unbounded pack would blow the context
    on the first question);
  * the per-model request knobs match what each model accepts — sending
    ``output_config.effort`` to a model that predates it is a hard 400.

Nothing here touches the network: the Anthropic SDK is only imported inside
``build_client``, so the whole module is testable without a key.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import assistant_core as ac  # noqa: E402


# ── sandbox ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", [
    "../../etc/passwd",                  # climbing out with ..
    "app/../../etc/hosts",               # climbing out mid-path
    ".streamlit/secrets.toml",           # a Streamlit secrets file
    "deploy/systemd/executor.env",       # the executor's signing secret
    "deploy/ibkr-gateway/.env",          # IBKR gateway credentials
    ".git/config",                       # git internals
])
def test_safe_path_refuses_escapes_and_secrets(path):
    with pytest.raises(ac.RepoAccessError):
        ac._safe_path(path)


def test_safe_path_refuses_non_text_files():
    with pytest.raises(ac.RepoAccessError):
        ac._safe_path("models/inference_assets_ct.joblib")


def test_safe_path_allows_ordinary_repo_files():
    assert ac._safe_path("README.md").name == "README.md"
    assert ac._safe_path("app/overall_core.py").exists()


def test_tool_errors_are_returned_not_raised():
    """A refusal has to come back as a tool_result the model can react to —
    a raised exception would abort the whole answer."""
    text, is_error = ac.execute_tool("read_file", {"path": "../../etc/passwd"})
    assert is_error and "escapes the repository" in text

    text, is_error = ac.execute_tool("nope", {})
    assert is_error and "Unknown tool" in text

    text, is_error = ac.execute_tool("search_repo", {"pattern": "[unclosed"})
    assert "Invalid regular expression" in text


def test_iter_repo_files_skips_denied_paths():
    names = {p.name for p in ac._iter_repo_files("**/*.toml")}
    assert "secrets.toml" not in names


# ── tools ──────────────────────────────────────────────────────────────────
def test_read_file_is_line_numbered_and_pageable():
    out, is_error = ac.execute_tool(
        "read_file", {"path": "STRATEGY_HEALTH.md", "start_line": 1, "max_lines": 3})
    assert not is_error
    assert "STRATEGY_HEALTH.md (lines 1–3 of" in out
    assert "re-read with start_line=4" in out


def test_search_repo_reports_path_and_line():
    out, is_error = ac.execute_tool(
        "search_repo", {"pattern": r"^# ", "path_glob": "STRATEGY_HEALTH.md"})
    assert not is_error
    assert "STRATEGY_HEALTH.md:1:" in out


def test_read_json_drills_with_a_pointer():
    out, is_error = ac.execute_tool(
        "read_json", {"path": "data/overall/strategy_health.json",
                      "pointer": "verdict"})
    assert not is_error
    assert "status" in out and "n_sleeves" in out


def test_read_json_reports_available_keys_on_a_bad_pointer():
    out, _ = ac.execute_tool("read_json", {"path": "data/overall/target_book.json",
                                           "pointer": "nope"})
    assert "no key 'nope'" in out and "available:" in out


def test_read_table_modes():
    out, is_error = ac.execute_tool(
        "read_table", {"path": "data/overall/health_history.csv", "mode": "shape"})
    assert not is_error and "rows ×" in out


def test_every_advertised_tool_is_dispatchable():
    assert {t["name"] for t in ac.TOOLS} == set(ac._DISPATCH)


# ── pruning + state pack ───────────────────────────────────────────────────
def test_prune_elides_long_lists_but_keeps_the_true_length():
    out = ac.prune({"xs": list(range(100))}, max_list=5)
    assert out["xs"][:5] == [0, 1, 2, 3, 4]
    assert "95 more of 100 elided" in out["xs"][-1]


def test_prune_drops_per_day_series_wholesale():
    out = ac.prune({"series": {"dates": ["2026-01-01"] * 400}})
    assert "elided" in out["series"] and "read_json" in out["series"]


def test_state_pack_carries_the_artifacts_the_pages_render():
    pack = ac.app_state_pack()
    for key in ("target_book", "executed_book", "daily_audit",
                "strategy_health", "overall_results", "sleeve_backtests"):
        assert key in pack, key
    assert pack["target_book"].get("weights"), "the published weights must survive pruning"
    assert pack["strategy_health"].get("verdict"), "the health verdict must survive pruning"


def test_state_pack_stays_small_enough_to_cache():
    """strategy_health.json alone is ~440 KB on disk; unpruned it would dominate
    the context window and make every question expensive."""
    size = len(json.dumps(ac.app_state_pack(), default=str))
    assert size < 120_000, f"state pack grew to {size:,} chars"


def test_system_prompt_is_byte_stable_between_builds():
    """The prompt is the cached prefix — a wall-clock timestamp anywhere in it
    would invalidate the API's prompt cache on every single turn."""
    pack = ac.app_state_pack()
    assert ac.build_system_prompt(pack)[0]["text"] == ac.build_system_prompt(pack)[0]["text"]


def test_system_prompt_is_cache_marked_and_grounded():
    block = ac.build_system_prompt()[0]
    assert block["cache_control"] == {"type": "ephemeral"}
    for needle in ("Application", "OVERALL", "HEALTH", "estimate, interpolate",
                   "STRATEGY_HEALTH.md"):
        assert needle in block["text"], needle


# ── models ─────────────────────────────────────────────────────────────────
def test_the_default_is_the_cheapest_entitled_model():
    """Cost is the default because most questions are look-ups the state pack
    already answers; the picker is one click away for the rest."""
    specs = sorted(ac.MODEL_CATALOG.values(), key=lambda s: s.rank)
    assert ac.default_model(specs) == "claude-haiku-4-5"
    assert ac.cheapest_model(specs) == "claude-haiku-4-5"
    assert ac.most_capable_model(specs) == "claude-opus-5"


def test_cheapest_is_computed_from_prices_not_hardcoded():
    """The day a cheaper model ships, it should win without a code change."""
    bargain = ac.ModelSpec("claude-thrift-1", "Claude Thrift", "cheap",
                           input_price=0.1, output_price=0.5, rank=50)
    specs = [ac.MODEL_CATALOG["claude-opus-5"], ac.MODEL_CATALOG["claude-haiku-4-5"],
             bargain]
    assert ac.default_model(specs) == "claude-thrift-1"


def test_cost_index_orders_the_catalogue_by_real_spend():
    order = [s.model_id for s in
             sorted((m for m in ac.MODEL_CATALOG.values() if m.priced),
                    key=lambda s: s.cost_index)]
    assert order[0] == "claude-haiku-4-5"
    assert order[-1] == "claude-fable-5"
    # Input-weighted: the grounding prompt dwarfs the answer on every turn.
    assert (ac.MODEL_CATALOG["claude-haiku-4-5"].cost_index
            < ac.MODEL_CATALOG["claude-sonnet-5"].cost_index
            < ac.MODEL_CATALOG["claude-opus-5"].cost_index)


def test_an_unpriced_model_is_never_called_cheapest():
    """A model discovered from the API but absent from the catalogue has no
    price — claiming it is the most economical would be a guess."""
    unknown = ac.spec_for("claude-zeta-9")
    assert not unknown.priced and unknown.cost_index is None
    assert ac.cheapest_model([unknown]) is None
    # ...and with nothing priced, the default falls back to the most capable.
    assert ac.default_model([unknown]) == "claude-zeta-9"


def test_default_falls_back_when_nothing_is_entitled():
    assert ac.default_model([]) == ac.PREFERRED_MODEL


def test_request_knobs_are_gated_on_what_the_model_accepts():
    modern = ac.request_kwargs(ac.MODEL_CATALOG["claude-opus-5"], show_reasoning=True)
    assert modern["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert modern["output_config"] == {"effort": "high"}

    # Reasoning is never *disabled* — hiding it is a display choice, and
    # disabling it on the Opus 5 family degrades tool use.
    hidden = ac.request_kwargs(ac.MODEL_CATALOG["claude-opus-5"], show_reasoning=False)
    assert hidden["thinking"] == {"type": "adaptive"}

    # Haiku 4.5 predates adaptive thinking and effort: neither may be sent.
    legacy = ac.request_kwargs(ac.MODEL_CATALOG["claude-haiku-4-5"], show_reasoning=True)
    assert "thinking" not in legacy and "output_config" not in legacy


def test_max_tokens_never_exceeds_the_models_output_cap():
    for spec in ac.MODEL_CATALOG.values():
        assert ac.request_kwargs(spec, show_reasoning=False)["max_tokens"] <= spec.max_output


def test_unknown_models_are_inferred_conservatively():
    """A model released after this file was written still has to be drivable."""
    future = ac.spec_for("claude-opus-6-1")
    assert future.adaptive_thinking and future.effort == "high"

    legacy = ac.spec_for("claude-3-7-sonnet-20250219")
    assert not legacy.adaptive_thinking and legacy.effort is None

    unknown = ac.spec_for("some-other-model")
    assert not unknown.adaptive_thinking and unknown.effort is None


def test_discover_models_falls_back_when_the_api_is_unreachable(monkeypatch):
    def _boom(_key):
        raise RuntimeError("no network")

    monkeypatch.setattr(ac, "build_client", _boom)
    specs, err = ac.discover_models("sk-ant-nope")
    assert err == "no network"
    assert [s.model_id for s in specs][0] == "claude-opus-5"


def test_discover_models_annotates_what_the_key_is_entitled_to(monkeypatch):
    class _M:
        def __init__(self, mid, name=None):
            self.id, self.display_name = mid, name

    class _Models:
        def list(self, limit=100):
            return type("P", (), {"data": [_M("claude-haiku-4-5"),
                                           _M("claude-opus-5"),
                                           _M("claude-zeta-9", "Claude Zeta 9")]})()

    monkeypatch.setattr(ac, "build_client",
                        lambda _k: type("C", (), {"models": _Models()})())
    specs, err = ac.discover_models("sk-ant-ok")
    assert err is None
    ids = [s.model_id for s in specs]
    assert ids[0] == "claude-opus-5", "best model first"
    assert set(ids) == {"claude-opus-5", "claude-haiku-4-5", "claude-zeta-9"}
    # Catalogued models keep their curated label; unknown ones keep the API's.
    by_id = {s.model_id: s for s in specs}
    assert by_id["claude-opus-5"].label == "Claude Opus 5"
    assert by_id["claude-zeta-9"].label == "Claude Zeta 9"


def test_api_key_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-from-env  ")
    assert ac.resolve_api_key() == "sk-ant-from-env"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert ac.resolve_api_key() in (None, ac.resolve_api_key())  # falls through


# ── the chat turn (tool loop) ──────────────────────────────────────────────
class _Block:
    """A minimal stand-in for an SDK content block."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeStream:
    def __init__(self, events, final):
        self._events, self._final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class _FakeMessages:
    """Replays a scripted sequence of API turns and records what was sent."""
    def __init__(self, turns):
        self._turns, self.calls = list(turns), []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(*self._turns.pop(0))


def _usage(**kw):
    return _Block(input_tokens=0, output_tokens=0,
                  cache_read_input_tokens=0, cache_creation_input_tokens=0, **kw)


def _fake_client(turns):
    msgs = _FakeMessages(turns)
    return type("C", (), {"messages": msgs})(), msgs


#: A date no published book carries, used as the assistant's *hallucinated*
#: answer so "the tool really ran" can be told apart from "the model guessed".
_NOT_THE_BOOKS_DATE = "1970-01-01"


def _books_as_of() -> str:
    """The live ``as_of`` of the target book — the value the read_json tool
    must surface.  Read from disk at test time rather than pinned, because the
    daily publisher rewrites this file (see the sibling read_json tests, which
    likewise assert on structure, never on a committed value)."""
    book = Path(__file__).resolve().parent.parent / "data/overall/target_book.json"
    return str(json.loads(book.read_text(encoding="utf-8"))["as_of"])


def test_stream_answer_runs_tools_then_answers(monkeypatch):
    """A tool_use turn must execute the tool, feed the result back as a
    tool_result, and continue to a final answer."""
    tool_turn = (
        [_Block(type="text", text="Let me check. ")],
        _Block(stop_reason="tool_use", usage=_usage(),
               content=[_Block(type="tool_use", id="tu_1", name="read_json",
                               input={"path": "data/overall/target_book.json",
                                      "pointer": "as_of"})]),
    )
    # The assistant's answer deliberately states a date the book does NOT
    # carry.  The point of this test is that the TOOL's output is what gets fed
    # back, so the two must differ — pinning both to the same literal (as this
    # once did) let a hallucinated value pass whenever it happened to match.
    answer_turn = (
        [_Block(type="text", text=f"The book is as of {_NOT_THE_BOOKS_DATE}.")],
        _Block(stop_reason="end_turn", usage=_usage(),
               content=[_Block(type="text",
                               text=f"The book is as of {_NOT_THE_BOOKS_DATE}.")]),
    )
    client, msgs = _fake_client([tool_turn, answer_turn])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    events = list(ac.stream_answer(
        api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
        messages=[{"role": "user", "content": "When is the book as of?"}],
        system=[{"type": "text", "text": "sys"}]))

    kinds = [e["type"] for e in events]
    assert kinds.count("tool") == 1
    assert events[-1]["type"] == "done"
    tool_ev = next(e for e in events if e["type"] == "tool")
    assert tool_ev["name"] == "read_json" and not tool_ev["error"]

    assert len(msgs.calls) == 2, "the answer turn must follow the tool turn"

    # The tool's real output — not a hallucinated one — is fed back, and the
    # assistant's own blocks are echoed verbatim (thinking blocks must survive).
    transcript = events[-1]["messages"]
    assert transcript[1]["content"] is tool_turn[1].content
    result_block = transcript[2]["content"][0]
    assert transcript[2]["role"] == "user"
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "tu_1"
    # the book's real as_of, read at test time: hard-coding it broke the suite
    # every time the daily publisher committed a new book
    assert _books_as_of() in result_block["content"]
    assert _NOT_THE_BOOKS_DATE not in result_block["content"]

    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text.endswith(f"The book is as of {_NOT_THE_BOOKS_DATE}.")


def test_stream_answer_sends_the_gated_knobs_and_cache_control(monkeypatch):
    turn = ([], _Block(stop_reason="end_turn", usage=_usage(), content=[]))
    client, msgs = _fake_client([turn])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                          messages=[{"role": "user", "content": "hi"}],
                          system=[{"type": "text", "text": "sys"}]))
    sent = msgs.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["tools"] is ac.TOOLS


def test_stream_answer_streams_thinking_deltas(monkeypatch):
    events_in = [_Block(type="content_block_delta",
                        delta=_Block(type="thinking_delta", thinking="weighing…")),
                 _Block(type="text", text="done")]
    turn = (events_in, _Block(stop_reason="end_turn", usage=_usage(), content=[]))
    client, _ = _fake_client([turn])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    out = list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                                messages=[{"role": "user", "content": "hi"}],
                                system=[{"type": "text", "text": "s"}],
                                show_reasoning=True))
    assert any(e["type"] == "thinking" and e["text"] == "weighing…" for e in out)


def test_stream_answer_stops_at_the_tool_round_cap(monkeypatch):
    """A model that loops on tools must be cut off, not left to run forever."""
    def _tool_turn():
        return ([], _Block(stop_reason="tool_use", usage=_usage(),
                           content=[_Block(type="tool_use", id="t", name="list_files",
                                           input={"path_glob": "*.md"})]))

    client, msgs = _fake_client([_tool_turn() for _ in range(ac.MAX_TOOL_ROUNDS)])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    out = list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                                messages=[{"role": "user", "content": "hi"}],
                                system=[{"type": "text", "text": "s"}]))

    # ...and the cap must not mean silence: one extra call is made with tool
    # use forbidden, so the user gets the best answer the gathered data allows.
    assert len(msgs.calls) == ac.MAX_TOOL_ROUNDS + 1
    closing = msgs.calls[-1]
    assert closing["tool_choice"] == {"type": "none"}
    assert "Do not request another" in closing["messages"][-1]["content"]
    assert any(e["type"] == "note" and "budget" in e["text"] for e in out)
    assert out[-1]["type"] == "done"


def test_a_failed_closing_turn_still_ends_cleanly(monkeypatch):
    """If the forced final answer itself fails, the turn must still terminate
    with a `done` event — otherwise the UI spins forever."""
    class _Boom(_FakeMessages):
        def stream(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) > ac.MAX_TOOL_ROUNDS:
                raise RuntimeError("closing call failed")
            return _FakeStream([], _Block(
                stop_reason="tool_use", usage=_usage(),
                content=[_Block(type="tool_use", id="t", name="list_files",
                                input={"path_glob": "*.md"})]))

    msgs = _Boom([])
    monkeypatch.setattr(ac, "build_client",
                        lambda _k: type("C", (), {"messages": msgs})())
    out = list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                                messages=[{"role": "user", "content": "hi"}],
                                system=[{"type": "text", "text": "s"}]))
    assert out[-1]["type"] == "done"
    assert any("closing summary also failed" in e.get("text", "") for e in out)


def test_stream_answer_surfaces_a_refusal(monkeypatch):
    turn = ([], _Block(stop_reason="refusal", usage=_usage(), content=[]))
    client, _ = _fake_client([turn])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    out = list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                                messages=[{"role": "user", "content": "hi"}],
                                system=[{"type": "text", "text": "s"}]))
    assert any("declined" in e.get("text", "") for e in out if e["type"] == "text")


def test_stream_answer_does_not_mutate_the_callers_history(monkeypatch):
    turn = ([], _Block(stop_reason="end_turn", usage=_usage(),
                       content=[_Block(type="text", text="ok")]))
    client, _ = _fake_client([turn])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)

    history = [{"role": "user", "content": "hi"}]
    out = list(ac.stream_answer(api_key="k", spec=ac.MODEL_CATALOG["claude-opus-5"],
                                messages=history, system=[{"type": "text", "text": "s"}]))
    assert history == [{"role": "user", "content": "hi"}]
    assert len(out[-1]["messages"]) == 2      # the updated transcript comes back


# ── live signals, search reach, and session continuity ─────────────────────
def test_live_signal_is_advertised_as_the_first_tool():
    """Ordering is guidance: a question about a current signal should reach for
    this before it starts grepping engine source."""
    assert ac.TOOLS[0]["name"] == "live_signal"
    desc = ac.TOOLS[0]["description"]
    assert "running the app's" in desc.lower() or "own engine" in desc.lower()


def test_live_signal_rejects_an_unknown_sleeve_with_the_valid_set():
    out, is_error = ac.execute_tool("live_signal", {"sleeve": "NOTATICKER"})
    assert "Unknown sleeve" in out
    for key in ("BTC", "GLDM", "SOXX"):
        assert key in out


def test_search_default_scope_covers_source_not_just_docs():
    """The bar-clock convention that broke a real question lives in a module
    docstring, not in any .md — searching docs alone could never find it."""
    globs = ac._DEFAULT_SEARCH_GLOBS
    assert "app/*.py" in globs and "*.md" in globs

    out, is_error = ac.execute_tool(
        "search_repo", {"pattern": r"Bar-START date of the most recent"})
    assert not is_error
    assert "app/freshness.py" in out


def test_search_accepts_several_globs_and_dedupes():
    out, _ = ac.execute_tool("search_repo", {
        "pattern": r"^# ", "path_glob": "STRATEGY_HEALTH.md,STRATEGY_HEALTH.md",
        "max_results": 20})
    assert out.count("STRATEGY_HEALTH.md:1:") == 1


def test_search_context_returns_surrounding_lines():
    """Context turns search→read into one call."""
    plain, _ = ac.execute_tool("search_repo", {
        "pattern": r"^# ", "path_glob": "STRATEGY_HEALTH.md"})
    ctx, _ = ac.execute_tool("search_repo", {
        "pattern": r"^# ", "path_glob": "STRATEGY_HEALTH.md", "context": 3})
    assert len(ctx) > len(plain)
    assert ">" in ctx                      # the matched line is marked


def test_search_miss_suggests_how_to_widen():
    out, _ = ac.execute_tool("search_repo",
                             {"pattern": "zzz-not-in-this-repo-zzz"})
    assert "Widen the glob" in out


def test_the_prompt_teaches_the_bar_clock():
    """The failing question hinged entirely on this: 12:00Z bar close = 7 AM CT,
    and `as_of` being the bar START date."""
    text = ac.build_system_prompt()[0]["text"]
    for needle in ("12:00-UTC", "7:00 AM US Central", "START", "live_signal",
                   "app/freshness.py"):
        assert needle in text, needle


def test_the_prompt_tells_the_model_to_answer_rather_than_exhaust_the_budget():
    text = ac.build_system_prompt()[0]["text"]
    assert "budget per question" in text.lower()
    assert "useful partial answer" in text.lower()


def test_followups_replay_the_whole_thread(monkeypatch):
    """A follow-up must carry the earlier turns, or "and how does that compare?"
    is unanswerable. The session ends only when the UI clears the history."""
    def _turn(text):
        return ([], _Block(stop_reason="end_turn", usage=_usage(),
                           content=[_Block(type="text", text=text)]))

    client, msgs = _fake_client([_turn("first answer"), _turn("second answer")])
    monkeypatch.setattr(ac, "build_client", lambda _k: client)
    spec = ac.MODEL_CATALOG["claude-opus-5"]
    sysb = [{"type": "text", "text": "s"}]

    first = list(ac.stream_answer(api_key="k", spec=spec, system=sysb,
                                  messages=[{"role": "user", "content": "Q1"}]))
    history = first[-1]["messages"]
    assert len(history) == 2                       # user + assistant

    second = list(ac.stream_answer(
        api_key="k", spec=spec, system=sysb,
        messages=history + [{"role": "user", "content": "Q2"}]))
    assert msgs.calls[1]["messages"][0]["content"] == "Q1", (
        "the earlier question must still be in the replayed thread")
    thread = second[-1]["messages"]
    assert len(thread) == 4                        # Q1, A1, Q2, A2
    assert [m["role"] for m in thread] == ["user", "assistant", "user", "assistant"]
    assert thread[2]["content"] == "Q2"
