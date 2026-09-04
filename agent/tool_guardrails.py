"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


# Kept on the module so a frozen Babel backend can verify that the imported
# guardrail implementation includes the cross-pagination failure circuit
# breaker, even when Python does not expose a useful module origin.
BABEL_HERMES_RUNTIME_CAPABILITIES = frozenset(
    {
        "same_target_failure",
        "terminal_batch_observation_guardrails",
    }
)


IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "cronjob",
        "delegate_task",
        "process",
    }
)


# A terminal tool call is a mutation-capable tool because its payload may run
# arbitrary shell code.  That classification is intentionally retained.  This
# narrower set is only used to identify repeated, read-only observations packed
# into one shell payload, which would otherwise bypass the per-tool-call
# failure circuit breaker.
_TERMINAL_OBSERVATION_COMMANDS = frozenset(
    {"cat", "file", "head", "ls", "readlink", "stat", "tail", "test", "wc"}
)
_TERMINAL_COMMAND_WRAPPERS = frozenset({"builtin", "command", "env", "sudo"})
_SHELL_SEGMENT_SEPARATOR = re.compile(r"(?:&&|\|\||[;|&])")


def _terminal_shell_segments(command: str) -> list[str]:
    """Split simple shell command chains without splitting quoted separators.

    This is deliberately a small inspection heuristic, not a shell parser. It
    only needs to recognize direct command chains such as ``stat path; stat
    path``. Quoted strings and escaped separators remain part of their command,
    so a legitimate argument containing ``;`` does not become a false repeat.
    Unsupported shell syntax simply produces no additional segments and is
    left to the normal terminal executor and guardrails.
    """

    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            current.append(character)
            escaped = True
            index += 1
            continue
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            index += 1
            continue

        separator = _SHELL_SEGMENT_SEPARATOR.match(command, index)
        if separator is not None:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index = separator.end()
            continue
        if character == "\n":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue
        current.append(character)
        index += 1

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def _terminal_observation_target(
    tokens: list[str],
) -> tuple[str, str] | None:
    """Return ``(command, target)`` for a conservative read-only invocation."""

    if not tokens:
        return None
    command_index = 0
    while command_index < len(tokens) and tokens[command_index] in _TERMINAL_COMMAND_WRAPPERS:
        command_index += 1
    if command_index >= len(tokens):
        return None

    executable = os.path.basename(tokens[command_index])
    if executable not in _TERMINAL_OBSERVATION_COMMANDS:
        return None
    arguments = tokens[command_index + 1 :]
    if not arguments:
        return None

    targets: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument == "]":
            continue
        if argument == "--":
            continue
        if executable == "test" and argument in {"!", "["}:
            continue
        if argument.startswith("-"):
            # Common option/value pairs for these inspection commands. Unknown
            # flags are ignored rather than guessed as paths.
            if argument in {
                "-c", "--format", "-f", "--printf", "-n", "--lines",
                "-m", "--max-count", "-t", "--time-style",
            }:
                skip_next = True
            continue
        targets.append(argument)

    # A command with multiple independent targets is not an exact repeated
    # observation of one target. It may be a legitimate inventory operation.
    if len(targets) != 1:
        return None
    target = os.path.normpath(targets[0])
    if not target or target == ".":
        return None
    return executable, target


def repeated_terminal_observation_target(
    command: str,
    *,
    minimum_repeats: int = 2,
) -> tuple[str, str, int] | None:
    """Find one repeated read-only command/target within a terminal payload.

    The return value is safe guardrail metadata: command name, normalized
    target, and repeat count. This does not execute or rewrite the shell
    payload. Distinct targets and distinct observation commands remain allowed.
    """

    if not isinstance(command, str) or not command.strip():
        return None
    try:
        threshold = max(2, int(minimum_repeats))
    except (TypeError, ValueError):
        threshold = 2

    counts: dict[tuple[str, str], int] = {}
    for segment in _terminal_shell_segments(command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            # Malformed quoting is not evidence of a repeated observation.
            continue
        observation = _terminal_observation_target(tokens)
        if observation is None:
            continue
        count = counts.get(observation, 0) + 1
        counts[observation] = count
    repeated = [
        (observation, count)
        for observation, count in counts.items()
        if count >= threshold
    ]
    if not repeated:
        return None
    (command_name, target), count = max(repeated, key=lambda item: item[1])
    return command_name, target, count


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
        )


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


_PAGINATION_ARGUMENTS = frozenset(
    {
        "cursor",
        "end_line",
        "line_end",
        "line_start",
        "limit",
        "max_bytes",
        "max_lines",
        "offset",
        "page",
        "start_line",
    }
)


def failure_target_signature(
    tool_name: str,
    args: Mapping[str, Any] | None,
) -> ToolCallSignature | None:
    """Return a stable identity for a failed idempotent target.

    Pagination changes the observation window, not the target. A missing file
    can therefore evade exact-call detection when a model retries the same
    path with a new ``offset`` on every turn. Successful pagination remains
    unconstrained; this identity is only counted by the failure path.
    """

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool not in IDEMPOTENT_TOOL_NAMES:
        return None
    stable_args = {
        str(key): value
        for key, value in _coerce_args(args).items()
        if str(key).strip().lower() not in _PAGINATION_ARGUMENTS
    }
    return ToolCallSignature.from_call(normalized_tool, stable_args)


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._same_target_failure_counts: dict[ToolCallSignature, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

        target_signature = failure_target_signature(tool_name, args)
        if target_signature is not None and target_signature != signature:
            target_count = self._same_target_failure_counts.get(target_signature, 0)
            if target_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="block",
                    code="same_target_failure_block",
                    message=(
                        f"Blocked {tool_name}: the same target has failed {target_count} "
                        "times despite pagination or observation changes. Stop retrying "
                        "the target unchanged; change strategy or explain the blocker."
                    ),
                    tool_name=tool_name,
                    count=target_count,
                    signature=target_signature,
                )
                self._halt_decision = decision
                return decision

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: this read-only call returned the same "
                            f"result {repeat_count} times. Stop repeating it unchanged; "
                            "use the result already provided or try a different query."
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            target_signature = failure_target_signature(tool_name, args)
            target_count = 0
            if target_signature is not None:
                target_count = self._same_target_failure_counts.get(target_signature, 0) + 1
                self._same_target_failure_counts[target_signature] = target_count

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            # Tool names are capabilities, not requests. Distinct reads may
            # legitimately miss several candidate paths before locating an
            # existing source file, and distinct compiler/test commands may
            # expose independent defects. Never terminate a turn from that
            # aggregate count. Exact failing arguments and identical
            # no-progress reads retain their separate hard stops.

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if (
                target_signature is not None
                and target_signature != signature
                and self.config.warnings_enabled
                and target_count >= self.config.exact_failure_warn_after
            ):
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_target_failure_warning",
                    message=(
                        f"{tool_name} has failed {target_count} times for the same target "
                        "despite pagination or observation changes. Inspect the error and "
                        "change strategy instead of retrying that target unchanged."
                    ),
                    tool_name=tool_name,
                    count=target_count,
                    signature=target_signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=(
                        f"{tool_name} has failed {same_count} times this turn. "
                        "This looks like a loop; change approach before retrying."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)
        target_signature = failure_target_signature(tool_name, args)
        if target_signature is not None:
            self._same_target_failure_counts.pop(target_signature, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
