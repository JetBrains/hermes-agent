"""Phase-2 tests for the junie-acp provider: recognition + interactive flow.

Covers what the runtime-path tests (tests/agent/test_junie_acp_client.py) do
not: that `hermes model` / setup machinery recognizes junie-acp and that the
interactive model flow writes the right config.
"""
from __future__ import annotations

import pytest

from hermes_cli import models as M
from hermes_cli import providers as P
from hermes_cli.provider_catalog import provider_catalog_by_slug


def test_junie_acp_is_canonical_provider():
    slugs = {e.slug for e in M.CANONICAL_PROVIDERS}
    assert "junie-acp" in slugs


def test_junie_acp_in_provider_catalog_with_label():
    by = provider_catalog_by_slug()
    assert "junie-acp" in by
    assert by["junie-acp"].label
    assert by["junie-acp"].description


@pytest.mark.parametrize("alias", ["junie", "jetbrains-junie-acp", "junie-acp-agent"])
def test_junie_aliases_resolve(alias):
    assert P.ALIASES.get(alias) == "junie-acp"
    full = P.resolve_provider_full(alias)
    assert full is not None


def test_junie_acp_curated_models_present():
    models = [m for m, _label in M.curated_models_for_provider("junie-acp")]
    assert "junie-acp" in models  # provider-default sentinel
    assert any(m.startswith("gemini") or m.startswith("claude") or m.startswith("gpt")
               for m in models)


def test_junie_acp_provider_model_ids():
    ids = M.provider_model_ids("junie-acp")
    assert "junie-acp" in ids


def test_junie_acp_flow_writes_config(monkeypatch):
    """The interactive flow persists provider/base_url/api_mode correctly."""
    import hermes_cli.auth as auth
    from hermes_cli.model_setup_flows import _model_flow_junie_acp
    from hermes_cli.config import load_config

    monkeypatch.setattr(
        auth, "get_external_process_provider_status",
        lambda pid: {"resolved_command": "/usr/bin/junie", "command": "junie",
                     "base_url": "acp://junie"},
    )
    monkeypatch.setattr(
        auth, "resolve_external_process_provider_credentials",
        lambda pid: {"base_url": "acp://junie", "command": "/usr/bin/junie",
                     "args": ["--acp=true"]},
    )
    monkeypatch.setattr(auth, "_prompt_model_selection", lambda *a, **k: "gemini-3-flash-preview")

    _model_flow_junie_acp({}, current_model="")

    cfg = load_config()
    assert isinstance(cfg["model"], dict)
    assert cfg["model"]["provider"] == "junie-acp"
    assert cfg["model"]["base_url"] == "acp://junie"
    assert cfg["model"]["api_mode"] == "chat_completions"


def test_junie_acp_flow_missing_cli_does_not_write(monkeypatch):
    """When the Junie CLI can't be resolved, the flow bails without writing."""
    import hermes_cli.auth as auth
    from hermes_cli.model_setup_flows import _model_flow_junie_acp
    from hermes_cli.config import load_config

    monkeypatch.setattr(
        auth, "get_external_process_provider_status",
        lambda pid: {"command": "junie", "base_url": "acp://junie"},
    )

    def _raise(pid):
        raise RuntimeError("Could not find the CLI command 'junie'")

    monkeypatch.setattr(auth, "resolve_external_process_provider_credentials", _raise)
    # Selection must never be reached.
    monkeypatch.setattr(
        auth, "_prompt_model_selection",
        lambda *a, **k: pytest.fail("should not prompt when CLI is missing"),
    )

    _model_flow_junie_acp({}, current_model="")

    cfg = load_config()
    model = cfg.get("model")
    if isinstance(model, dict):
        assert model.get("provider") != "junie-acp"
