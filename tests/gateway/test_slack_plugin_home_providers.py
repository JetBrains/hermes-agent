"""Tests for plugin-registered Slack App Home providers.

Covers:
* ``PluginContext.register_slack_home_provider`` validation + queuing
* ``PluginManager.get_slack_home_providers`` accessor + unload cleanup
* ``SlackAdapter._handle_app_home_opened`` dispatch for ``tab == "home"``
* Non-home tab preservation (messages / other)
* Provider failure isolation (Socket Mode dispatch continues)
* Deferred loading / plugin reload (providers looked up at event time)
* Multi-workspace client routing via team_id
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Mock slack-bolt so SlackAdapter can be imported even without the package
# ---------------------------------------------------------------------------

def _ensure_slack_mock() -> None:
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    """Build a fresh PluginManager + PluginContext bound to it."""
    mgr = PluginManager()
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        description="test",
    )
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


def _make_adapter() -> SlackAdapter:
    config = PlatformConfig(enabled=True, token="xoxb-fake")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._team_clients = {}
    adapter._team_bot_user_ids = {}
    return adapter


# ---------------------------------------------------------------------------
# PluginContext.register_slack_home_provider — validation + queuing
# ---------------------------------------------------------------------------

class TestRegisterSlackHomeProviderAPI:
    """Behaviour of ctx.register_slack_home_provider()."""

    def test_provider_is_queued(self):
        mgr, ctx = _make_ctx()

        async def provider(*, publish_home, **_):  # pragma: no cover
            await publish_home({"type": "home", "blocks": []})

        ctx.register_slack_home_provider(provider)

        providers = mgr.get_slack_home_providers()
        assert len(providers) == 1
        cb, plugin_name = providers[0]
        assert cb is provider
        assert plugin_name == "test_plugin"

    def test_non_callable_raises(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_slack_home_provider("not-a-callable")  # type: ignore[arg-type]

    def test_registration_handle_releases_provider(self):
        mgr, ctx = _make_ctx()

        async def provider(**_):  # pragma: no cover
            return None

        handle = ctx.register_slack_home_provider(provider)
        assert len(mgr.get_slack_home_providers()) == 1
        handle.dispose()
        assert mgr.get_slack_home_providers() == []

    def test_unload_clears_providers(self):
        mgr, ctx = _make_ctx()

        async def provider(**_):  # pragma: no cover
            return None

        ctx.register_slack_home_provider(provider)
        assert mgr.get_slack_home_providers()
        mgr.unload()
        assert mgr.get_slack_home_providers() == []


# ---------------------------------------------------------------------------
# SlackAdapter._handle_app_home_opened — home dispatch + tab preservation
# ---------------------------------------------------------------------------

class TestSlackHomeProviderDispatch:
    """Runtime dispatch path for app_home_opened."""

    @pytest.mark.asyncio
    async def test_home_tab_invokes_provider_and_publish_home(self):
        adapter = _make_adapter()
        team_client = AsyncMock()
        team_client.views_publish = AsyncMock(return_value={"ok": True})
        adapter._team_clients["T_WS"] = team_client

        calls: list[dict] = []

        async def provider(*, client, user_id, team_id, publish_home, event, body, **_):
            calls.append(
                {
                    "client": client,
                    "user_id": user_id,
                    "team_id": team_id,
                    "event": event,
                    "body": body,
                }
            )
            await publish_home(
                {
                    "type": "home",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": "hi"},
                        }
                    ],
                }
            )

        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [(provider, "dash")]

        event = {
            "type": "app_home_opened",
            "tab": "home",
            "user": "U_USER",
        }
        body = {"team_id": "T_WS", "event": event}

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(event, body)

        assert len(calls) == 1
        assert calls[0]["client"] is team_client
        assert calls[0]["user_id"] == "U_USER"
        assert calls[0]["team_id"] == "T_WS"
        assert calls[0]["body"]["team_id"] == "T_WS"
        team_client.views_publish.assert_awaited_once_with(
            user_id="U_USER",
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "hi"},
                    }
                ],
            },
        )

    @pytest.mark.asyncio
    async def test_messages_tab_does_not_invoke_provider(self):
        adapter = _make_adapter()
        adapter._set_assistant_suggested_prompts = AsyncMock()
        adapter._seed_agent_dm_session = MagicMock()
        adapter._cache_agent_view_context = MagicMock()

        provider = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [(provider, "dash")]

        event = {
            "type": "app_home_opened",
            "tab": "messages",
            "team": "T_TEAM",
            "channel": "D123",
            "user": "U_USER",
        }

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(event)

        provider.assert_not_awaited()
        # Existing messages-tab side effects still run.
        adapter._seed_agent_dm_session.assert_called_once()
        adapter._set_assistant_suggested_prompts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_other_tab_is_noop(self):
        adapter = _make_adapter()
        adapter._set_assistant_suggested_prompts = AsyncMock()
        provider = AsyncMock()
        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [(provider, "dash")]

        event = {"type": "app_home_opened", "tab": "about", "user": "U_USER"}

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(event)

        provider.assert_not_awaited()
        adapter._set_assistant_suggested_prompts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_failure_is_isolated(self):
        adapter = _make_adapter()
        adapter._team_clients["T1"] = AsyncMock()

        order: list[str] = []

        async def boom(**_):
            order.append("boom")
            raise RuntimeError("provider exploded")

        async def ok(*, publish_home, **_):
            order.append("ok")
            await publish_home({"type": "home", "blocks": []})

        client = adapter._team_clients["T1"]
        client.views_publish = AsyncMock(return_value={"ok": True})

        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [
            (boom, "bad_plugin"),
            (ok, "good_plugin"),
        ]

        event = {"type": "app_home_opened", "tab": "home", "user": "U1"}
        body = {"team_id": "T1"}

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            # Must not raise — Socket Mode dispatch continues.
            await adapter._handle_app_home_opened(event, body)

        assert order == ["boom", "ok"]
        client.views_publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plugin_manager_failure_does_not_break_dispatch(self):
        adapter = _make_adapter()
        event = {"type": "app_home_opened", "tab": "home", "user": "U1"}

        with patch(
            "hermes_cli.plugins.get_plugin_manager",
            side_effect=RuntimeError("plugins broken"),
        ):
            await adapter._handle_app_home_opened(event)  # must not raise

    @pytest.mark.asyncio
    async def test_deferred_lookup_sees_provider_registered_after_connect(self):
        """Providers are read at event time — reload/deferred load works."""
        adapter = _make_adapter()
        client = AsyncMock()
        client.views_publish = AsyncMock(return_value={"ok": True})
        adapter._team_clients["T1"] = client

        # Simulate "no providers at connect time".
        live_providers: list = []
        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.side_effect = lambda: list(live_providers)

        event = {"type": "app_home_opened", "tab": "home", "user": "U1"}
        body = {"team_id": "T1"}

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(event, body)
            client.views_publish.assert_not_awaited()

            # Plugin registers later (reload / deferred load) — no reconnect.
            async def provider(*, publish_home, **_):
                await publish_home({"type": "home", "blocks": []})

            live_providers.append((provider, "late_plugin"))
            await adapter._handle_app_home_opened(event, body)

        client.views_publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_workspace_routes_to_correct_client(self):
        adapter = _make_adapter()
        client_a = AsyncMock()
        client_a.views_publish = AsyncMock(return_value={"ok": True})
        client_b = AsyncMock()
        client_b.views_publish = AsyncMock(return_value={"ok": True})
        adapter._team_clients = {"T_A": client_a, "T_B": client_b}
        # Primary fallback must not be used when team_id matches.
        adapter._app.client.views_publish = AsyncMock()

        seen_clients: list = []

        async def provider(*, client, publish_home, **_):
            seen_clients.append(client)
            await publish_home({"type": "home", "blocks": []})

        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [(provider, "dash")]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(
                {"type": "app_home_opened", "tab": "home", "user": "U1"},
                {"team_id": "T_B"},
            )
            await adapter._handle_app_home_opened(
                {"type": "app_home_opened", "tab": "home", "user": "U2"},
                {"team_id": "T_A"},
            )

        assert seen_clients == [client_b, client_a]
        client_b.views_publish.assert_awaited_once_with(
            user_id="U1",
            view={"type": "home", "blocks": []},
        )
        client_a.views_publish.assert_awaited_once_with(
            user_id="U2",
            view={"type": "home", "blocks": []},
        )
        adapter._app.client.views_publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_narrow_signature_receives_only_declared_fields(self):
        """Additive payload: narrow callbacks don't get unexpected kwargs."""
        adapter = _make_adapter()
        adapter._team_clients["T1"] = AsyncMock()

        received: dict = {}

        async def narrow(*, user_id, team_id):
            received["user_id"] = user_id
            received["team_id"] = team_id

        fake_mgr = MagicMock()
        fake_mgr.get_slack_home_providers.return_value = [(narrow, "dash")]

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr):
            await adapter._handle_app_home_opened(
                {"type": "app_home_opened", "tab": "home", "user": "U9"},
                {"team_id": "T1"},
            )

        assert received == {"user_id": "U9", "team_id": "T1"}
