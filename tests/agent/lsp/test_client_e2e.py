"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(
    workspace: Path,
    script: str = "clean",
    extra_env: dict[str, str] | None = None,
) -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    if extra_env:
        env.update(extra_env)
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_didchange_bumps_version(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        f.write_text("print('hi 2')\n")
        v1 = await client.open_file(str(f), language_id="python")  # re-open path = didChange
        assert v1 == v0 + 1
        await client.wait_for_diagnostics(str(f), v1, mode="document")
        # Mock pushed a diagnostic for both events; merged view has one
        # entry (push store keyed by file path).
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_handles_crashing_server(tmp_path: Path):
    """When the server exits right after initialize, subsequent requests
    fail gracefully (not hang)."""
    f = tmp_path / "x.py"
    f.write_text("")

    client = _client(tmp_path, "crash")
    await client.start()  # should succeed (mock answers initialize before crashing)
    # Give the OS a moment to deliver the EOF.
    await asyncio.sleep(0.2)
    # The reader loop should detect EOF and mark pending requests as failed.
    try:
        await asyncio.wait_for(
            client.open_file(str(f), language_id="python"), timeout=2.0
        )
    except Exception:
        pass  # any exception is acceptable; the contract is "doesn't hang"
    await client.shutdown()


@pytest.mark.asyncio
async def test_client_shutdown_idempotent(tmp_path: Path):
    """Calling shutdown twice must be safe."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "clean")
    await client.start()
    await client.shutdown()
    await client.shutdown()  # must not raise


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-only")
@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
async def test_client_shutdown_terminates_descendant_processes(tmp_path: Path):
    child_pid_file = tmp_path / "child.pid"
    client = _client(
        tmp_path,
        "orphan-child",
        {"MOCK_LSP_CHILD_PID_FILE": str(child_pid_file)},
    )
    child_pid = None
    await client.start()
    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if child_pid_file.exists():
                child_pid = int(child_pid_file.read_text())
                break
            await asyncio.sleep(0.02)
        assert child_pid is not None
        os.kill(child_pid, 0)

        await client.shutdown()
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("LSP descendant survived client shutdown")
    finally:
        await client.shutdown()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_client_diagnostics_are_deduped(tmp_path: Path):
    """Repeated identical pushes must not produce duplicate diagnostics."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "errors")
    await client.start()
    try:
        for _ in range(3):
            v = await client.open_file(str(f), language_id="python")
            await client.wait_for_diagnostics(str(f), v, mode="document")
        diags = client.diagnostics_for(str(f))
        # Push store overwrites on every notification — should have 1.
        assert len(diags) == 1
    finally:
        await client.shutdown()
