"""Focused tests for API server communications + costs summary endpoints."""

import time
from datetime import datetime, timedelta, timezone

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_get("/api/communications", adapter._handle_list_communications)
    app.router.add_get("/api/costs/summary", adapter._handle_costs_summary)
    return app


def _seed_gateway_session(
    db: SessionDB,
    session_id: str,
    *,
    source: str = "slack",
    user_id: str = "U1",
    chat_id: str = "C1",
    chat_type: str = "channel",
    display_name: str = "Alice",
    thread_id: str | None = None,
    message_count: int = 3,
    estimated_cost_usd: float | None = 0.1,
    actual_cost_usd: float | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    started_at: float | None = None,
    end: bool = False,
    system_prompt: str = "SECRET SYSTEM PROMPT",
    origin_json: str = '{"token":"do-not-leak","chat_id":"C1"}',
):
    session_key = f"{source}:{chat_id}:{user_id}" + (f":{thread_id}" if thread_id else "")
    db.create_session(
        session_id,
        source,
        user_id=user_id,
        session_key=session_key,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
        display_name=display_name,
        system_prompt=system_prompt,
        origin_json=origin_json,
    )
    started = time.time() if started_at is None else started_at
    with db._lock:
        db._conn.execute(
            """
            UPDATE sessions SET
                message_count = ?,
                estimated_cost_usd = ?,
                actual_cost_usd = ?,
                input_tokens = ?,
                output_tokens = ?,
                started_at = ?,
                system_prompt = ?,
                origin_json = ?
            WHERE id = ?
            """,
            (
                message_count,
                estimated_cost_usd,
                actual_cost_usd,
                input_tokens,
                output_tokens,
                started,
                system_prompt,
                origin_json,
                session_id,
            ),
        )
        db._conn.commit()
    if end:
        db.end_session(session_id, "user_ended")
    return session_id


@pytest.mark.asyncio
async def test_capabilities_advertises_communications_and_costs(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    endpoints = data["endpoints"]
    assert endpoints["communications"] == {"method": "GET", "path": "/api/communications"}
    assert endpoints["costs_summary"] == {"method": "GET", "path": "/api/costs/summary"}
    # Existing session list contract remains advertised unchanged.
    assert endpoints["sessions"] == {"method": "GET", "path": "/api/sessions"}


@pytest.mark.asyncio
async def test_communications_requires_auth(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/communications")
        assert resp.status == 401

        ok = await cli.get(
            "/api/communications",
            headers={"Authorization": "Bearer sk-test"},
        )
        assert ok.status == 200


@pytest.mark.asyncio
async def test_costs_summary_requires_auth(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/costs/summary")
        assert resp.status == 401

        ok = await cli.get(
            "/api/costs/summary?days=7",
            headers={"Authorization": "Bearer sk-test"},
        )
        assert ok.status == 200


@pytest.mark.asyncio
async def test_communications_sanitizes_and_lists_active_and_finished(adapter, session_db):
    _seed_gateway_session(
        session_db,
        "active-1",
        user_id="U-active",
        chat_id="C-general",
        chat_type="channel",
        display_name="Active User",
        message_count=7,
        end=False,
    )
    _seed_gateway_session(
        session_db,
        "ended-1",
        user_id="U-ended",
        chat_id="D-dm",
        chat_type="im",
        display_name="Finished User",
        message_count=2,
        end=True,
    )
    # Non-gateway / non-slack noise should not appear when filtering source=slack
    session_db.create_session("cli-only", "cli")

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/communications?source=slack")
        assert resp.status == 200
        payload = await resp.json()

        active_only = await cli.get("/api/communications?source=slack&active_only=true")
        assert active_only.status == 200
        active_payload = await active_only.json()

        # Existing /api/sessions projection must stay free of gateway routing fields.
        sessions_resp = await cli.get("/api/sessions?source=slack")
        assert sessions_resp.status == 200
        sessions_payload = await sessions_resp.json()

    assert payload["object"] == "list"
    assert payload["source"] == "slack"
    assert payload["active_only"] is False
    assert len(payload["data"]) == 2

    by_id = {row["id"]: row for row in payload["data"]}
    active = by_id["active-1"]
    ended = by_id["ended-1"]

    assert active["active"] is True
    assert active["ended_at"] is None
    assert active["user_id"] == "U-active"
    assert active["display_name"] == "Active User"
    assert active["chat_id"] == "C-general"
    assert active["chat_type"] == "channel"
    assert active["message_count"] == 7
    assert active["source"] == "slack"
    assert active["session_key"]

    assert ended["active"] is False
    assert ended["ended_at"] is not None
    assert ended["chat_type"] == "im"
    assert ended["display_name"] == "Finished User"

    forbidden = {
        "origin_json",
        "system_prompt",
        "model_config",
        "has_system_prompt",
        "token",
        "SECRET SYSTEM PROMPT",
    }
    for row in payload["data"]:
        assert forbidden.isdisjoint(row.keys())
        dumped = str(row)
        assert "SECRET SYSTEM PROMPT" not in dumped
        assert "do-not-leak" not in dumped
        assert "origin_json" not in dumped

    assert active_payload["active_only"] is True
    assert [row["id"] for row in active_payload["data"]] == ["active-1"]

    # Client-safe sessions list still omits gateway routing metadata.
    for session in sessions_payload["data"]:
        assert "session_key" not in session
        assert "chat_id" not in session
        assert "chat_type" not in session
        assert "display_name" not in session
        assert "origin_json" not in session


@pytest.mark.asyncio
async def test_communications_empty_db(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/communications?source=slack")
        assert resp.status == 200
        payload = await resp.json()
    assert payload == {
        "object": "list",
        "data": [],
        "source": "slack",
        "active_only": False,
    }


@pytest.mark.asyncio
async def test_costs_summary_empty_db_is_honest(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/costs/summary?days=7")
        assert resp.status == 200
        payload = await resp.json()

    assert payload["object"] == "hermes.costs.summary"
    assert payload["days"] == 7
    assert payload["daily"] == []
    assert payload["last_day"] is None
    assert payload["last_week"] is None
    assert payload["totals"] is None
    assert payload["basis"] == "session_started_at_day"
    assert payload["currency"] == "USD"
    assert "Best-effort" in payload["note"]
    # Never present fabricated measured spend.
    assert payload.get("last_day") is None


@pytest.mark.asyncio
async def test_costs_summary_daily_and_rollups(adapter, session_db):
    now = time.time()
    today = datetime.now(timezone.utc).date()
    three_days_ago = datetime(
        today.year, today.month, today.day, tzinfo=timezone.utc
    ) - timedelta(days=3)

    # Today: actual cost preferred path has both actual + estimated
    _seed_gateway_session(
        session_db,
        "cost-today",
        user_id="U1",
        chat_id="C1",
        message_count=1,
        estimated_cost_usd=0.20,
        actual_cost_usd=0.15,
        input_tokens=100,
        output_tokens=40,
        started_at=now,
    )
    # Older day inside last week: estimated only
    _seed_gateway_session(
        session_db,
        "cost-old",
        user_id="U2",
        chat_id="C2",
        message_count=4,
        estimated_cost_usd=0.05,
        actual_cost_usd=None,
        input_tokens=20,
        output_tokens=10,
        started_at=three_days_ago.timestamp(),
    )
    # Outside default 7-day window
    _seed_gateway_session(
        session_db,
        "cost-ancient",
        user_id="U3",
        chat_id="C3",
        estimated_cost_usd=9.99,
        actual_cost_usd=9.99,
        input_tokens=999,
        output_tokens=999,
        started_at=now - (20 * 86400),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/costs/summary?days=7")
        assert resp.status == 200
        payload = await resp.json()

        narrow = await cli.get("/api/costs/summary?days=1")
        assert narrow.status == 200
        narrow_payload = await narrow.json()

    assert payload["basis"] == "session_started_at_day"
    assert payload["days"] == 7
    assert len(payload["daily"]) == 2

    days = {row["day"] for row in payload["daily"]}
    assert today.isoformat() in days
    assert three_days_ago.date().isoformat() in days

    # last_day is the most recent day bucket (today)
    assert payload["last_day"]["day"] == today.isoformat()
    assert payload["last_day"]["actual_cost_usd"] == pytest.approx(0.15)
    assert payload["last_day"]["estimated_cost_usd"] == pytest.approx(0.20)
    assert payload["last_day"]["sessions"] == 1

    assert payload["last_week"] is not None
    assert payload["last_week"]["sessions"] == 2
    assert payload["last_week"]["estimated_cost_usd"] == pytest.approx(0.25)
    # actual only present on one day → rollup keeps actual sum
    assert payload["last_week"]["actual_cost_usd"] == pytest.approx(0.15)
    assert payload["last_week"]["input_tokens"] == 120
    assert payload["last_week"]["output_tokens"] == 50

    assert payload["totals"]["sessions"] == 2
    # Ancient session excluded from 7-day window
    assert all(row["estimated_cost_usd"] != pytest.approx(9.99) for row in payload["daily"])

    assert narrow_payload["days"] == 1
    assert len(narrow_payload["daily"]) == 1
    assert narrow_payload["daily"][0]["day"] == today.isoformat()
