"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(
    *tool_names: str,
    max_iterations: int = 10,
    config: dict | None = None,
    **agent_kwargs,
) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **agent_kwargs,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_action_board_context_budget_can_use_compact_recovery_floor():
    scoped = _make_agent(
        "read_file",
        request_context_budget_tokens=1_024,
        babel_action_board_scoped=True,
    )
    interactive = _make_agent(
        "read_file",
        request_context_budget_tokens=1_024,
    )

    assert scoped.request_context_budget_tokens == 2_048
    assert interactive.request_context_budget_tokens == 8_192


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_action_board_mutation_first_gate_reopens_tools_after_real_edit():
    """One targeted read is allowed; inventory waits until a mutation lands."""

    agent = _make_agent("read_file", "write_file", "terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_first_required = True
    messages = []

    pre_mutation_calls = [
        _mock_tool_call("read_file", '{"path":"package.json"}', "c-read-allowed"),
        _mock_tool_call(
            "write_file",
            '{"path":"src/index.ts","content":"broken"}',
            "c-write-failed",
        ),
    ]
    post_mutation_calls = [
        _mock_tool_call(
            "write_file",
            '{"path":"src/index.ts","content":"export {};"}',
            "c-write",
        ),
        _mock_tool_call("terminal", '{"command":"npm test"}', "c-terminal-allowed"),
    ]

    with patch(
        "run_agent.handle_function_call",
        side_effect=[
            json.dumps({"content": "{}"}),
            json.dumps({"error": "write rejected"}),
            json.dumps({"bytes_written": 10}),
            json.dumps({"output": "tests passed", "exit_code": 0}),
        ],
    ) as mock_hfc:
        for call in pre_mutation_calls:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-mutation-first",
            )
        assert agent._scoped_terminal_progress_seen is False
        for call in post_mutation_calls:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-mutation-first",
            )

    assert mock_hfc.call_count == 4
    by_id = {message["tool_call_id"]: message["content"] for message in messages}
    assert "tests passed" in by_id["c-terminal-allowed"]
    assert agent._scoped_terminal_progress_seen is True


def test_action_board_mutation_first_gate_halts_repeated_policy_rejection():
    agent = _make_agent("terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_first_required = True
    messages = []

    with patch("run_agent.handle_function_call") as mock_hfc:
        for index in range(2):
            call = _mock_tool_call(
                "terminal",
                '{"command":"ls -la"}',
                f"c-blocked-{index}",
            )
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-mutation-first-stop",
            )

    mock_hfc.assert_not_called()
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "scoped_mutation_first_repeat"
    assert agent._tool_guardrail_halt_decision.count == 2


def test_action_board_read_only_verifier_allows_inventory_and_exact_test():
    inventory_agent = _make_agent("terminal")
    inventory_agent._babel_scoped_worker = True
    inventory_agent.persist_tool_guardrails_across_turns = True
    inventory_agent._scoped_read_only_recovery = True
    inventory_messages = []

    inventory = _mock_tool_call(
        "terminal",
        '{"command":"ls -la"}',
        "c-verifier-inventory",
    )
    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"output": "clean", "exit_code": 0}),
    ) as inventory_hfc:
        inventory_agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[inventory]),
            inventory_messages,
            "task-verifier",
        )

    inventory_hfc.assert_called_once()
    assert inventory_agent._tool_guardrail_halt_decision is None
    assert "clean" in inventory_messages[0]["content"]

    exact_agent = _make_agent("terminal")
    exact_agent._babel_scoped_worker = True
    exact_agent.persist_tool_guardrails_across_turns = True
    exact_agent._scoped_read_only_recovery = True
    exact_agent._scoped_exact_terminal_command = "npm test -- --runInBand"
    exact_messages = []
    exact_test = _mock_tool_call(
        "terminal",
        '{"command":"npm test -- --runInBand"}',
        "c-verifier-test",
    )
    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"output": "tests passed", "exit_code": 0}),
    ) as exact_hfc:
        exact_agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[exact_test]),
            exact_messages,
            "task-verifier",
        )
    exact_hfc.assert_called_once()
    assert "tests passed" in exact_messages[0]["content"]


def test_read_only_verifier_allows_distinct_evidence_batch_without_mutation():
    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True

    for index in range(4):
        agent._append_guardrail_observation(
            "read_file",
            {"path": f"src/evidence-{index}.ts"},
            json.dumps({"content": f"evidence {index}"}),
            failed=False,
        )

    assert agent._tool_guardrail_halt_decision is None


def test_read_only_verifier_halts_second_read_of_same_path():
    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True

    for _index in range(2):
        agent._append_guardrail_observation(
            "read_file",
            {"path": "package.json"},
            json.dumps({"content": "{}"}),
            failed=False,
        )

    assert agent._tool_guardrail_halt_decision is not None
    assert (
        agent._tool_guardrail_halt_decision.code
        == "scoped_verification_repeat_path"
    )


def test_read_only_verifier_allows_multiple_distinct_exact_terminal_commands():
    agent = _make_agent("terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True

    for index in range(2):
        agent._append_guardrail_observation(
            "terminal",
            {"command": f"npm test -- --run evidence-{index}"},
            json.dumps({"output": "checked", "exit_code": 0}),
            failed=False,
        )
        block_message = agent._scoped_read_only_recovery_block_message(
            "terminal",
            {"command": f"npm test -- --run evidence-{index}"},
        )
        if index == 0:
            assert block_message is None

    assert agent._tool_guardrail_halt_decision is None
    assert agent._scoped_terminal_inventory_calls == 0


def test_read_only_verifier_halts_when_exact_command_changes():
    agent = _make_agent("terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True
    agent._scoped_exact_terminal_command = "npm test"

    message = agent._scoped_read_only_recovery_block_message(
        "terminal",
        {"command": "cat package.json"},
    )

    assert message is not None
    assert agent._tool_guardrail_halt_decision is not None
    assert (
        agent._tool_guardrail_halt_decision.code
        == "scoped_verification_command_mismatch"
    )


def test_read_only_exact_verification_pass_is_self_terminating():
    agent = _make_agent("terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True
    agent._scoped_exact_terminal_command = "npm test"

    agent._append_guardrail_observation(
        "terminal",
        {"command": "npm test"},
        json.dumps({"output": "tests passed", "exit_code": 0}),
        failed=False,
    )

    assert agent._scoped_verification_terminal_result == {
        "status": "passed",
        "command": "npm test",
        "exit_code": 0,
    }
    assert agent._tool_guardrail_halt_decision is None


def test_read_only_exact_verification_failure_routes_without_second_call():
    agent = _make_agent("terminal")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_read_only_recovery = True
    agent._scoped_exact_terminal_command = "npm test"

    agent._append_guardrail_observation(
        "terminal",
        {"command": "npm test"},
        json.dumps({"output": "tests failed", "exit_code": 1}),
        failed=True,
    )

    assert agent._scoped_verification_terminal_result == {
        "status": "failed",
        "command": "npm test",
        "exit_code": 1,
    }
    assert agent._tool_guardrail_halt_decision is not None
    assert (
        agent._tool_guardrail_halt_decision.code
        == "scoped_verification_command_failed"
    )


def test_action_board_mutation_first_gate_allows_distinct_grounding_reads_and_blocks_rereads():
    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_first_required = True
    messages = []
    calls = [
        _mock_tool_call("read_file", '{"path":"package.json"}', "c-read-one"),
        _mock_tool_call("read_file", '{"path":".gitignore"}', "c-read-two"),
        _mock_tool_call("read_file", '{"path":"tsconfig.json"}', "c-read-three"),
        _mock_tool_call("read_file", '{"path":"vite.config.ts"}', "c-read-four"),
    ]

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"content": "{}"}),
    ) as mock_hfc:
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=calls),
            messages,
            "task-mutation-first-parallel",
        )

    assert mock_hfc.call_count == 4
    by_id = {message["tool_call_id"]: message["content"] for message in messages}
    assert "content" in by_id["c-read-one"]
    assert "content" in by_id["c-read-two"]
    assert "content" in by_id["c-read-three"]
    assert "content" in by_id["c-read-four"]
    assert agent._tool_guardrail_halt_decision is None

    # A duplicate path is corrected once and halted only when the provider
    # ignores that exact correction, while new targeted paths remain useful.
    for index in range(2):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        "read_file",
                        '{"path":"package.json"}',
                        f"c-read-repeat-{index}",
                    )
                ],
            ),
            messages,
            "task-mutation-first-parallel",
        )
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "scoped_mutation_first_repeat"


def test_babel_action_board_prompt_enables_persistent_hard_stop_without_private_kwargs():
    """The host boundary must survive adapters that drop scoped kwargs."""

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("search_files")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.deepseek.com/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            ephemeral_system_prompt="Action Board execution boundary: worktree-local only",
        )

    assert agent.persist_tool_guardrails_across_turns is True
    assert agent._tool_guardrails.config.hard_stop_enabled is True
    assert agent._tool_guardrails.config.no_progress_block_after == 5


def test_action_board_user_rule_marker_enables_dispatch_hard_stop():
    """The scoped circuit breaker survives a pooled prompt refresh."""

    agent = _make_agent("search_files")
    agent.ephemeral_system_prompt = ""
    messages = [
        {
            "role": "user",
            "content": "ACTION BOARD EXECUTION RULE: inspect once, then edit.",
        }
    ]
    result = json.dumps({"total_count": 1, "files": ["./package.json"]})

    with patch("run_agent.handle_function_call", return_value=result) as mock_hfc:
        for index in range(7):
            agent._execute_tool_calls_sequential(
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        _mock_tool_call(
                            "search_files",
                            '{"pattern":"*","path":"."}',
                            f"c-{index}",
                        )
                    ],
                ),
                messages,
                "task-1",
            )

    assert agent.persist_tool_guardrails_across_turns is True
    assert agent._tool_guardrails.config.hard_stop_enabled is True
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code in {
        "idempotent_no_progress_block",
        "scoped_exact_repeat_block",
    }
    assert mock_hfc.call_count == 5


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_hard_stop_dispatch_serializes_repeated_paginated_target_before_execution():
    """A repeated missing target cannot escape through one parallel batch."""

    config = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 99,
                "same_tool_failure": 3,
                "idempotent_no_progress": 99,
            },
        }
    }
    agent = _make_agent("read_file", config=config)
    calls = [
        _mock_tool_call(
            "read_file",
            json.dumps({"path": "missing.md", "offset": offset, "limit": 1}),
            f"c-{offset}",
        )
        for offset in range(1, 7)
    ]
    messages = []

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"error": "File not found"}),
    ) as mock_hfc:
        agent._execute_tool_calls(
            SimpleNamespace(content="", tool_calls=calls),
            messages,
            "task-repeated-paginated-target",
        )

    # The first three observations are allowed. The fourth is blocked before
    # dispatch, and the remaining speculative calls are skipped in order.
    assert mock_hfc.call_count == 3
    assert len(messages) == len(calls)
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "same_target_failure_block"
    assert "same_target_failure_block" in messages[3]["content"]
    assert "skipped" in messages[4]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.get_pre_tool_call_block_message", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)


def test_config_enabled_hard_stop_run_conversation_returns_controlled_guardrail_halt_without_top_level_error():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 2
    assert result["api_calls"] == 3
    assert result["api_calls"] < agent.max_iterations
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "error" not in result
    assert result["completed"] is True
    assert "stopped retrying" in result["final_response"]
    assert result["guardrail"]["code"] == "repeated_exact_failure_block"
    assert result["guardrail"]["tool_name"] == "web_search"

    assistant_tool_calls = [
        m for m in result["messages"]
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    for assistant_msg in assistant_tool_calls:
        call_ids = [tc["id"] for tc in assistant_msg["tool_calls"]]
        following_results = [
            m for m in result["messages"]
            if m.get("role") == "tool" and m.get("tool_call_id") in call_ids
        ]
        assert len(following_results) == len(call_ids)


def test_exact_verification_run_completes_after_one_model_and_tool_call():
    agent = _make_agent(
        "terminal",
        max_iterations=10,
        babel_action_board_scoped=True,
    )
    exact_command = "npm test"
    agent.client.chat.completions.create.return_value = _mock_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            _mock_tool_call(
                "terminal",
                json.dumps({"command": exact_command}),
                "c-exact-verification",
            )
        ],
    )
    prompt = (
        "BABEL_CONTINUATION_READ_ONLY_RECOVERY: 1\n"
        "BABEL_CONTINUATION_EXACT_TERMINAL_COMMAND_JSON: "
        + json.dumps(exact_command)
    )

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"output": "tests passed", "exit_code": 0}),
        ) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(prompt)

    mock_hfc.assert_called_once()
    assert agent.client.chat.completions.create.call_count == 1
    assert result["api_calls"] == 1
    assert result["turn_exit_reason"] == "scoped_verification_completed"
    assert result["completed"] is True
    assert result["final_response"].startswith(
        "Controller-owned verification passed with exit code 0."
    )
    assert result["scoped_verification"] == {
        "status": "passed",
        "command": exact_command,
        "exit_code": 0,
    }
    assert "guardrail" not in result


def test_exact_verification_failure_stops_before_another_model_or_tool_call():
    agent = _make_agent(
        "terminal",
        max_iterations=10,
        babel_action_board_scoped=True,
    )
    exact_command = "npm run build"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": exact_command}),
                    "c-exact-failed",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "cat package.json"}),
                    "c-inventory-must-not-run",
                )
            ],
        ),
    ]
    prompt = (
        "BABEL_CONTINUATION_READ_ONLY_RECOVERY: 1\n"
        "BABEL_CONTINUATION_EXACT_TERMINAL_COMMAND_JSON: "
        + json.dumps(exact_command)
    )

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"output": "missing build.js", "exit_code": 1}),
        ) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(prompt)

    mock_hfc.assert_called_once()
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["code"] == "scoped_verification_command_failed"
    assert result["scoped_verification"] == {
        "status": "failed",
        "command": exact_command,
        "exit_code": 1,
    }


def test_action_board_mutation_recovery_allows_progressive_single_path_reconciliation():
    """Changed writes to one manifest are progress; identical loops are fenced elsewhere."""

    agent = _make_agent("write_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    messages = []
    calls = [
        _mock_tool_call(
            "write_file",
            json.dumps({"path": "package.json", "content": f"{{\"revision\":{i}}}"}),
            f"c-{i}",
        )
        for i in range(1, 5)
    ]

    with patch("run_agent.handle_function_call", return_value=json.dumps({"bytes_written": 20})):
        for call in calls:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-recovery",
            )

    assert agent._tool_guardrail_halt_decision is None


def test_action_board_mutation_recovery_respects_disabled_path():
    agent = _make_agent("write_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    agent._scoped_disabled_mutation_paths = {"package.json"}
    messages = []
    call = _mock_tool_call(
        "write_file",
        json.dumps({"path": "package.json", "content": "{}"}),
        "c-disabled",
    )

    with patch("run_agent.handle_function_call", return_value=json.dumps({"bytes_written": 2})) as mock_hfc:
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[call]),
            messages,
            "task-recovery",
        )

    mock_hfc.assert_not_called()
    assert messages and "recovery mutation path is disabled" in messages[0]["content"]


def test_action_board_mutation_recovery_allows_distinct_grounding_reads():
    """A coherent repair may inspect several different contract files."""

    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    messages = []
    calls = [
        _mock_tool_call(
            "read_file",
            json.dumps({"path": path}),
            f"read-{index}",
        )
        for index, path in enumerate(
            ("package.json", "tsconfig.json", "vite.config.ts"),
            start=1,
        )
    ]

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"content": "grounding evidence"}),
    ) as mock_hfc:
        for call in calls:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-recovery-reads",
            )

    assert mock_hfc.call_count == 3
    assert agent._tool_guardrail_halt_decision is None


def test_action_board_mutation_recovery_has_no_aggregate_distinct_read_limit():
    """Distinct causal reads remain available; repeated paths carry the loop fence."""

    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    agent._scoped_recovery_read_limit_override = 1

    agent._append_guardrail_observation(
        "read_file",
        {"path": "server/index.ts"},
        json.dumps({"content": "server"}),
        failed=False,
    )
    assert agent._tool_guardrail_halt_decision is None

    agent._append_guardrail_observation(
        "read_file",
        {"path": "tsconfig.json"},
        json.dumps({"content": "config"}),
        failed=False,
    )

    assert agent._tool_guardrail_halt_decision is None


def test_action_board_mutation_recovery_halts_repeated_grounding_path():
    """One dedup redirect may recover; a third identical read halts the loop."""

    agent = _make_agent("read_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    messages = []
    calls = [
        _mock_tool_call(
            "read_file",
            json.dumps({"path": "package.json"}),
            f"repeat-{index}",
        )
        for index in range(1, 4)
    ]

    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"content": "same evidence"}),
    ):
        for index, call in enumerate(calls, start=1):
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-recovery-repeat",
            )
            if index == 2:
                assert agent._tool_guardrail_halt_decision is None

    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "scoped_recovery_repeat_path"
    assert "package.json" in agent._tool_guardrail_halt_decision.message
    assert "after 3 calls" in agent._tool_guardrail_halt_decision.message


def test_action_board_mutation_recovery_respects_disabled_dotfile_path():
    """Dotfiles keep their leading dot across recovery path normalization."""

    agent = _make_agent("write_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    agent._scoped_disabled_mutation_paths = {".env.example"}
    messages = []
    call = _mock_tool_call(
        "write_file",
        json.dumps({"path": ".env.example", "content": "PORT=4179\n"}),
        "c-disabled-dotfile",
    )

    with patch("run_agent.handle_function_call", return_value=json.dumps({"bytes_written": 11})) as mock_hfc:
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[call]),
            messages,
            "task-recovery-dotfile",
        )

    mock_hfc.assert_not_called()
    assert messages and "recovery mutation path is disabled" in messages[0]["content"]


def test_action_board_mutation_recovery_allows_progressive_patch_reconciliation():
    """V4A patches may refine one target while each call still changes evidence."""

    agent = _make_agent("patch")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    messages = []
    calls = [
        _mock_tool_call(
            "patch",
            json.dumps(
                {
                    "mode": "patch",
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Update File: package.json\n"
                        f"@@\n-{{\"revision\":{i - 1}}}\n+{{\"revision\":{i}}}\n"
                        "*** End Patch"
                    ),
                }
            ),
            f"patch-{i}",
        )
        for i in range(1, 5)
    ]

    with patch("run_agent.handle_function_call", return_value=json.dumps({"success": True})):
        for call in calls:
            agent._execute_tool_calls_sequential(
                SimpleNamespace(content="", tool_calls=[call]),
                messages,
                "task-recovery-patch",
            )

    assert agent._tool_guardrail_halt_decision is None


def test_action_board_mutation_recovery_matches_absolute_worktree_path():
    agent = _make_agent("write_file")
    agent._babel_scoped_worker = True
    agent.persist_tool_guardrails_across_turns = True
    agent._scoped_mutation_recovery = True
    agent._scoped_disabled_mutation_paths = {"server/index.ts"}
    messages = []
    call = _mock_tool_call(
        "write_file",
        json.dumps(
            {
                "path": (
                    "/private/tmp/run/checkouts/.babel-worktrees/card-123/"
                    "server/index.ts"
                ),
                "content": "export {};",
            }
        ),
        "c-absolute-disabled",
    )

    with patch("run_agent.handle_function_call", return_value=json.dumps({"bytes_written": 11})) as mock_hfc:
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[call]),
            messages,
            "task-recovery-absolute",
        )

    mock_hfc.assert_not_called()
    assert messages and "recovery mutation path is disabled" in messages[0]["content"]
