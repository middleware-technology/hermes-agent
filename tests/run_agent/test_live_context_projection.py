from __future__ import annotations

from agent.model_metadata import estimate_request_tokens_rough
from run_agent import AIAgent


def _agent(budget: int | None) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.request_context_budget_tokens = budget
    agent.tools = []
    agent.context_compressor = None
    return agent


def _long_history() -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "initial task"},
    ]
    for index in range(16):
        call_id = f"call-{index}"
        messages.extend([
            {
                "role": "assistant",
                "content": "reasoning " + "x" * 900,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"content":"' + "y" * 1200 + '"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "output " + "z" * 1800,
            },
        ])
    messages.append({"role": "user", "content": "finish the task"})
    return messages


def test_projection_is_disabled_without_budget() -> None:
    messages = _long_history()
    agent = _agent(None)
    projected = agent._project_live_context_for_request(messages)

    assert projected is messages
    assert agent._last_live_context_projection is None


def test_projection_compacts_provider_copy_and_preserves_durable_history() -> None:
    messages = _long_history()
    before = estimate_request_tokens_rough(messages, tools=[])
    agent = _agent(1_800)

    projected = agent._project_live_context_for_request(messages)
    after = estimate_request_tokens_rough(projected, tools=[])

    assert after <= 1_800
    assert before > after
    assert messages[-1]["content"] == "finish the task"
    assert projected[-1]["content"] == "finish the task"
    assert agent._last_live_context_projection["within_budget"] is True
    assert agent._last_live_context_projection["dropped_message_count"] > 0
    # Tool-call arguments remain valid JSON after compaction.
    for message in projected:
        for call in message.get("tool_calls", []):
            import json

            json.loads(call["function"]["arguments"])


def test_projection_drops_tool_call_groups_atomically_without_unavailable_stubs() -> None:
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "task " + "x" * 4200},
    ]
    for index in range(2):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "inspect",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "call_id": call_id,
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "terminal",
                    "content": "result " + "z" * 2200,
                },
            ]
        )

    agent = _agent(1_800)
    projected = agent._project_live_context_for_request(messages)

    assistant_calls = {
        call["id"]
        for message in projected
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    }
    result_ids = {
        message.get("tool_call_id")
        for message in projected
        if message.get("role") == "tool"
    }
    assert result_ids <= assistant_calls
    assert all(
        message.get("content") != "[Result unavailable — see context summary above]"
        for message in projected
    )
    # The most recent tool group is the only useful continuation state and
    # should survive when the budget cannot retain the entire transcript.
    assert "call-1" in assistant_calls
    assert "call-1" in result_ids


def test_projection_preserves_action_ledger_when_tool_history_is_dropped() -> None:
    """A bounded worker remembers landed files instead of restarting its card."""

    agent = _agent(8_192)
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "bounded file mutation " + "s" * 9_000,
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    original_instruction = (
        "ACTION BOARD FIRST-ACTION CONTRACT: continue concrete work.\n"
        + "policy " * 320
        + "\nCard: Establish project foundation\n"
        "Objective: Create package.json, tsconfig.json, and the server entrypoint.\n"
        "Acceptance Criteria: npm test, typecheck, and build pass.\n"
        + "durable audit context " * 650
    )
    messages = [
        {"role": "system", "content": "stable boundary " + "b" * 4_000},
        {"role": "user", "content": original_instruction},
    ]
    for index, path in enumerate(
        ("package.json", "tsconfig.json", "src/server/index.ts")
    ):
        call_id = f"write-{index}"
        messages.extend([
            {
                "role": "assistant",
                "content": f"Creating {path}",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": (
                            '{"path":"' + path + '","content":"'
                            + "x" * 2_400
                            + '"}'
                        ),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": '{"bytes_written":2400,"success":true}',
            },
        ])

    projected = agent._project_live_context_for_request(messages)
    rendered = "\n".join(str(item.get("content") or "") for item in projected)

    assert estimate_request_tokens_rough(projected, tools=agent.tools) <= 8_192
    assert "Card: Establish project foundation" in rendered
    assert "BABEL LIVE EXECUTION LEDGER" in rendered
    assert "write_file package.json" in rendered
    assert "write_file tsconfig.json" in rendered
    assert "write_file src/server/index.ts" in rendered
    assert "Do not restart" in rendered
    assert agent._last_live_context_projection["execution_ledger_applied"] is True
    assert agent._last_live_context_projection["projected_message_count"] > 2
