"""OpenAI-compatible shim that forwards Hermes requests to `junie --acp=true`.

This adapter lets Hermes treat the JetBrains Junie CLI's ACP server as a
chat-style backend. Each request starts a short-lived ACP session, sends the
formatted conversation as a single prompt, collects text chunks, and converts
the result back into the minimal shape Hermes expects from an OpenAI client.

It is a deliberate, standalone fork of ``agent/copilot_acp_client.py`` — kept
independent so upstream changes to the Copilot path cannot break Junie.

Junie specifics (vs Copilot):
  * launched as ``junie --acp=true`` (Copilot uses ``copilot --acp --stdio``);
  * auth is supplied via ``--auth <token>`` / ``JUNIE_API_KEY`` rather than
    being wholly owned by the CLI;
  * Junie wraps each JSON-RPC message with an extra
    ``"type": "com.agentclientprotocol.rpc.*"`` envelope field, which is
    harmless — messages are matched by ``id`` / ``method`` as usual.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.file_safety import get_read_block_error, is_write_denied
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://junie"
_DEFAULT_TIMEOUT_SECONDS = 900.0

# Junie's ACP session/update kinds that report tool activity. Unlike an OpenAI
# model, Junie is an autonomous agent that EXECUTES its own tools and reports
# them here (status pending -> in_progress -> completed/failed) — these are NOT
# delegation requests for Hermes to run. We consume them for observability, not
# to fabricate OpenAI tool_calls (see JunieACPClient._create_chat_completion).
_TOOL_UPDATE_KINDS = ("tool_call", "tool_call_update")


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_JUNIE_ACP_COMMAND", "").strip()
        or os.getenv("JUNIE_CLI_PATH", "").strip()
        or "junie"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_JUNIE_ACP_ARGS", "").strip()
    args = shlex.split(raw) if raw else ["--acp=true", "--skip-update-check"]
    # Inject auth from the environment when the caller hasn't already supplied
    # a token via --auth. Junie accepts a JetBrains/Junie token (perm-...).
    if "--auth" not in args and not any(a.startswith("--auth=") for a in args):
        token = os.getenv("JUNIE_API_KEY", "").strip()
        if token:
            args = args + ["--auth", token]
    return args


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    # Junie is an autonomous coding agent: it runs its OWN tools (read/edit/
    # execute) inside the ACP session and reports them via native tool_call
    # notifications. We therefore do NOT ask it to emit OpenAI-style tool
    # calls; it should just do the work and answer. Hermes' own tool schemas
    # are irrelevant to Junie's execution, so we don't inject them.
    del tools, tool_choice  # accepted for OpenAI-client compatibility; unused
    sections: list[str] = [
        "You are being used as the active ACP coding agent backend for Hermes.",
        "Use your own tools to complete the task, then answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _tool_update_text(update: dict[str, Any]) -> str:
    """Best-effort plain-text extraction from an ACP tool_call content list."""
    parts: list[str] = []
    for block in update.get("content") or []:
        if not isinstance(block, dict):
            continue
        inner = block.get("content")
        if isinstance(inner, dict) and isinstance(inner.get("text"), str):
            parts.append(inner["text"])
        elif isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(p for p in parts if p and p.strip()).strip()


def _merge_tool_update(store: dict[str, dict[str, Any]], update: dict[str, Any]) -> None:
    """Fold a tool_call / tool_call_update notification into ``store`` by id.

    Junie streams a ``tool_call`` (first-seen) then zero or more
    ``tool_call_update`` messages sharing the same ``toolCallId`` as the tool
    progresses (pending -> in_progress -> completed/failed). We keep the latest
    non-empty value for each field so ``store`` ends up with the final state.
    """
    tcid = str(update.get("toolCallId") or f"tool_{len(store)}")
    entry = store.setdefault(tcid, {"id": tcid})
    for field in ("title", "kind", "status"):
        val = update.get(field)
        if val:
            entry[field] = val
    text = _tool_update_text(update)
    if text:
        entry["result"] = text
    locations = update.get("locations")
    if locations:
        entry["locations"] = locations


def _render_tool_activity(tool_events: dict[str, dict[str, Any]]) -> str:
    """Render captured tool activity as a compact, human-readable summary.

    Surfaced via the assistant message's ``reasoning`` so the operator can see
    what Junie actually did, without misrepresenting completed actions as
    OpenAI tool_calls Hermes must execute.
    """
    lines: list[str] = []
    for ev in tool_events.values():
        kind = ev.get("kind", "tool")
        status = ev.get("status", "")
        title = ev.get("title", "")
        head = f"[{kind}] {title}".strip()
        if status:
            head = f"{head} ({status})"
        lines.append(head)
        result = ev.get("result")
        if result:
            snippet = result if len(result) <= 500 else result[:500] + "…"
            lines.append(f"    → {snippet}")
    return "\n".join(lines).strip()


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "JunieACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "JunieACPClient"):
        self.completions = _ACPChatCompletions(client)


class JunieACPClient:
    """Minimal OpenAI-client-compatible facade for JetBrains Junie ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "junie-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text, tool_events = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
        )

        # Junie executes its own tools and reports them as completed activity;
        # they are NOT delegation requests, so we never surface them as OpenAI
        # tool_calls (that would make Hermes try to re-run finished work).
        # Instead we log them and fold a readable summary into `reasoning`.
        activity = _render_tool_activity(tool_events)
        if tool_events:
            for ev in tool_events.values():
                logger.info(
                    "Junie ACP tool activity: kind=%s status=%s title=%s",
                    ev.get("kind"), ev.get("status"), ev.get("title"),
                )
        combined_reasoning = "\n\n".join(
            p for p in (
                (f"Junie tool activity:\n{activity}" if activity else ""),
                reasoning_text or "",
            ) if p
        ).strip() or None

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=response_text.strip(),
            tool_calls=[],
            reasoning=combined_reasoning,
            reasoning_content=combined_reasoning,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "junie-acp",
        )

    def _run_prompt(
        self, prompt_text: str, *, timeout_seconds: float
    ) -> tuple[str, str, dict[str, dict[str, Any]]]:
        try:
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Junie ACP command '{self._acp_command}'. "
                "Install the JetBrains Junie CLI or set HERMES_JUNIE_ACP_COMMAND/JUNIE_CLI_PATH."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Junie ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None, tool_events: dict[str, dict[str, Any]] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    tool_events=tool_events,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"Junie ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(f"Junie ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for Junie ACP response to {method}.")

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": True,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Junie ACP did not return a sessionId.")

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_events: dict[str, dict[str, Any]] = {}
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                tool_events=tool_events,
            )
            return "".join(text_parts), "".join(reasoning_parts), tool_events
        finally:
            self.close()

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        tool_events: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            if kind in _TOOL_UPDATE_KINDS:
                # Native structured tool activity (content is a list of blocks).
                # Captured for observability, NOT turned into OpenAI tool_calls.
                if tool_events is not None:
                    _merge_tool_update(tool_events, update)
                return True
            # agent_message_chunk / agent_thought_chunk carry a dict content.
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text()
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                if is_write_denied(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' is a protected system/credential file."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""))
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
