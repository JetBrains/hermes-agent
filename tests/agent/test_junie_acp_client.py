"""Tests for the JetBrains Junie ACP client shim.

Mirrors tests/agent/test_copilot_acp_client.py for the safety-critical
fs/permission bridge, plus Junie-specific command/args/auth resolution.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.junie_acp_client import JunieACPClient, _resolve_args, _resolve_command


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class JunieACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = JunieACPClient(acp_cwd="/tmp")

    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)

    def test_request_permission_is_not_auto_allowed(self) -> None:
        response = self._dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}},
            cwd="/tmp",
        )
        outcome = (((response.get("result") or {}).get("outcome") or {}).get("outcome"))
        self.assertEqual(outcome, "cancelled")

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")
            with patch("agent.redact._REDACT_ENABLED", True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "fs/read_text_file",
                        "params": {"path": str(secret_file)},
                    },
                    cwd=str(root),
                )
        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=", content)

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"
            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "fs/write_text_file",
                        "params": {"path": str(outside), "content": "should-not-write"},
                    },
                    cwd=str(root),
                )
        self.assertIn("error", response)
        self.assertFalse(outside.exists())


class JunieLaunchResolutionTests(unittest.TestCase):
    def test_default_command_and_args(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_command(), "junie")
            self.assertEqual(_resolve_args(), ["--acp=true", "--skip-update-check"])

    def test_command_override(self) -> None:
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_COMMAND": "/opt/junie"}, clear=True):
            self.assertEqual(_resolve_command(), "/opt/junie")
        with patch.dict(os.environ, {"JUNIE_CLI_PATH": "/usr/bin/junie"}, clear=True):
            self.assertEqual(_resolve_command(), "/usr/bin/junie")

    def test_auth_injected_from_env(self) -> None:
        with patch.dict(os.environ, {"JUNIE_API_KEY": "perm-token"}, clear=True):
            args = _resolve_args()
        self.assertIn("--auth", args)
        self.assertEqual(args[args.index("--auth") + 1], "perm-token")

    def test_explicit_auth_arg_not_double_injected(self) -> None:
        with patch.dict(
            os.environ,
            {"HERMES_JUNIE_ACP_ARGS": "--acp=true --auth explicit", "JUNIE_API_KEY": "perm-token"},
            clear=True,
        ):
            args = _resolve_args()
        self.assertEqual(args.count("--auth"), 1)
        self.assertIn("explicit", args)
        self.assertNotIn("perm-token", args)


if __name__ == "__main__":
    unittest.main()
