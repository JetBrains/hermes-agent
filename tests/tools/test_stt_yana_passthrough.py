"""End-to-end tests for consuming a yana-rendered ``config.stt`` block.

Context: the Junie Live (yana) config renderer copies
``agent-settings.hermess-agent.stt`` into Hermes' ``config.stt`` so a deployment
can select a speech-to-text backend (e.g. Whisper via the ingrazzio OpenAI proxy)
without an image rebuild. These tests pin the Hermes *consumption* side of that
contract — the half the renderer can't test — at two fidelity levels:

  Level 1 — config consumption (no audio, no network):
    the rendered block loads, selects the ``openai`` backend, and resolves to the
    exact (api_key, base_url) the OpenAI audio client will use.

  Level 2 — real transcription round-trip against a loopback stub:
    ``transcribe_audio`` drives the OpenAI SDK against a local HTTP server standing
    in for the proxy, proving the request path (/audio/transcriptions), the
    ``Authorization: Bearer <token>`` header, and the model actually reach the wire
    and the transcript comes back — decoupled from ingrazzio.
"""
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import transcription_tools as tt

# A block shaped exactly like render_config.py emits for
# stt.openai.provider = ingrazzio/openai (base_url ends in /v1; api_key = run token).
RENDERED_STT = {
    "enabled": True,
    "provider": "openai",
    "openai": {
        "model": "whisper-1",
        "base_url": "http://llm-proxy:8081/ingrazzio/tool/openai/v1",
        "api_key": "run-token-123",
    },
}


def _use_stt_config(monkeypatch, stt_block):
    """Point Hermes' config loader at ``stt_block`` (what the renderer produced).

    ``_load_stt_config`` does ``from hermes_cli.config import load_config`` at call
    time, so patching the attribute on that module is enough — and it also covers
    ``_resolve_openai_audio_client_config``, which loads the config the same way.
    """
    import hermes_cli.config as hconfig

    monkeypatch.setattr(hconfig, "load_config", lambda *a, **k: {"stt": stt_block})


def _write_valid_wav(path):
    """Write a tiny but structurally-valid WAV (0.1s silence) that passes
    ``_validate_audio_file`` and is a real multipart upload for the SDK."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)


# ── Level 1: config consumption ─────────────────────────────────────────────────
def test_rendered_stt_block_loads(monkeypatch):
    _use_stt_config(monkeypatch, RENDERED_STT)
    assert tt._load_stt_config() == RENDERED_STT
    assert tt.is_stt_enabled() is True


def test_rendered_stt_block_resolves_openai_client_config(monkeypatch):
    """The api_key/base_url the OpenAI audio client will use come straight from the
    rendered block — this is the exact seam render_config.py feeds."""
    _use_stt_config(monkeypatch, RENDERED_STT)
    api_key, base_url = tt._resolve_openai_audio_client_config()
    assert api_key == "run-token-123"
    assert base_url == "http://llm-proxy:8081/ingrazzio/tool/openai/v1"


def test_rendered_stt_block_selects_openai_provider(monkeypatch):
    """Explicit provider: openai + a resolvable key routes to the openai backend
    (no silent fallback to local, which is absent in the sandbox image)."""
    _use_stt_config(monkeypatch, RENDERED_STT)
    monkeypatch.setattr(tt, "_HAS_OPENAI", True)  # SDK present in the built image
    assert tt._get_provider(RENDERED_STT) == "openai"


# ── Level 2: real transcription round-trip against a loopback stub ────────────────
def test_openai_transcription_roundtrip_against_stub(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    if not tt._HAS_OPENAI:
        pytest.skip("openai SDK not installed")

    captured = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            length = int(self.headers.get("Content-Length", "0"))
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("Authorization")
            captured["body"] = self.rfile.read(length)
            body = b"hello from stub"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence the stub's stderr access log
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stt = {
            "enabled": True,
            "provider": "openai",
            "openai": {
                "model": "whisper-1",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "run-token-123",
            },
        }
        _use_stt_config(monkeypatch, stt)
        monkeypatch.setattr(tt, "_HAS_OPENAI", True)

        wav = tmp_path / "sample.wav"
        _write_valid_wav(wav)

        result = tt.transcribe_audio(str(wav))
    finally:
        server.shutdown()
        thread.join(timeout=5)

    # The whole hermes STT path executed against the wire:
    assert result["success"] is True, result
    assert result["transcript"] == "hello from stub"
    assert result["provider"] == "openai"
    # ...and the request reached the OpenAI audio endpoint with the right auth/model.
    assert captured["path"].endswith("/audio/transcriptions")
    assert captured["authorization"] == "Bearer run-token-123"
    assert b"whisper-1" in captured["body"]
