import asyncio
import pytest

from pathlib import Path
from types import SimpleNamespace
from hermes_cli import kanban_db as kb
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Allow the kanban notifier path-validator to upload artifacts the
    # tests write under ``tmp_path``. Without this, every artifact-delivery
    # test silently drops files because ``tmp_path`` isn't inside the
    # default ``MEDIA_DELIVERY_SAFE_ROOTS`` cache dirs.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    return home


def _assert_inherited_notify_sub(subs: list[dict]) -> None:
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "topic1"
    assert subs[0]["user_id"] == "user1"
    assert subs[0]["notifier_profile"] == "default"


def test_notify_sub_delivery_mode_persists_and_last_write_wins(kanban_home):
    """delivery_mode persists; an explicit re-subscribe is last-write-wins, a
    ``None`` re-subscribe leaves the existing mode untouched, an unknown value
    is ignored, and none of this clobbers the notifier_profile owner."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="mode sub task", assignee="worker1")
        # Fresh sub without a mode -> defaults to "notify".
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="owner-a",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["delivery_mode"] == "notify"
        assert subs[0]["notifier_profile"] == "owner-a"

        # Explicit re-subscribe changes the mode (last-write-wins) and must NOT
        # overwrite the existing owner (owner self-heals only when unset).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="owner-b", delivery_mode="wake",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["delivery_mode"] == "wake"
        assert subs[0]["notifier_profile"] == "owner-a"

        # A None re-subscribe leaves the existing mode untouched.
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["delivery_mode"] == "wake"

        # An unknown mode is ignored (treated like None: no clobber).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="bogus",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["delivery_mode"] == "wake"
    finally:
        conn.close()


def test_child_task_inherits_parent_delivery_mode(kanban_home):
    """Graph children inherit the parent's ACK edge AND its delivery_mode."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="root", assignee=None)
        kb.add_notify_sub(
            conn, task_id=parent, platform="telegram", chat_id="chat1",
            thread_id="42", user_id="u1", notifier_profile="default",
            delivery_mode="notify+wake",
        )
        child = kb.create_task(
            conn, title="review child", assignee="ccreviewer", parents=[parent],
        )
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "42"
    assert subs[0]["user_id"] == "u1"
    assert subs[0]["notifier_profile"] == "default"
    assert subs[0]["delivery_mode"] == "notify+wake"


def test_notify_sub_chat_type_persists_and_last_write_wins(kanban_home):
    """chat_type persists, defaults to 'dm', an explicit re-subscribe is
    last-write-wins, and a None re-subscribe leaves it untouched. The
    active-wake path replays this field so the woken turn keys to the
    operator's real channel."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="chat_type sub", assignee="worker1")
        # Fresh sub without chat_type -> defaults to "dm".
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "dm"

        # Explicit re-subscribe corrects the recorded chat_type (last-write-wins).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            chat_type="group",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "group"

        # A None re-subscribe (here changing only the mode) must NOT clobber
        # the recorded chat_type.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="wake",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "group"
        assert subs[0]["delivery_mode"] == "wake"
    finally:
        conn.close()


def test_child_task_inherits_parent_chat_type(kanban_home):
    """Graph children inherit the parent's chat_type alongside its ACK edge and
    delivery_mode, so a woken child notification keys to the same session as
    the parent's originating channel."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="root", assignee=None)
        kb.add_notify_sub(
            conn, task_id=parent, platform="telegram", chat_id="chat1",
            user_id="u1", chat_type="group", delivery_mode="notify+wake",
        )
        child = kb.create_task(
            conn, title="impl child", assignee="coder", parents=[parent],
        )
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["chat_type"] == "group"
    assert subs[0]["delivery_mode"] == "notify+wake"
    assert subs[0]["user_id"] == "u1"


@pytest.mark.asyncio
async def test_notifier_notify_plus_wake_sends_and_wakes(kanban_home):
    """notify+wake delivers the passive message AND wakes the agent; a plain
    notify sub only sends. The agent is woken only for the notify+wake sub."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        passive_tid = kb.create_task(conn, title="passive task", assignee="worker1")
        active_tid = kb.create_task(conn, title="active task", assignee="worker1")
        kb.add_notify_sub(conn, task_id=passive_tid, platform="telegram", chat_id="chat1")
        kb.add_notify_sub(
            conn, task_id=active_tid, platform="telegram", chat_id="chat1",
            delivery_mode="notify+wake",
        )
        kb.block_task(conn, passive_tid, reason="passive block")
        kb.block_task(conn, active_tid, reason="active block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    # Legacy/unstamped subs are only visible to the dispatch-owning
    # gateway (include_unowned); mimic that ownership like the
    # upstream notifier tests do.
    runner._kanban_dispatcher_lock_handle = object()

    fake_adapter = MagicMock()
    sent_msgs: list[str] = []

    async def _send(chat_id, msg, metadata=None):
        sent_msgs.append(msg)

    fake_adapter.send = AsyncMock(side_effect=_send)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    trigger_mock = AsyncMock(return_value={"triggered_agent": True})
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("tools.send_message_tool._trigger_gateway_agent", new=trigger_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Both subs still get a passive send (notify AND notify+wake send).
    assert len(sent_msgs) == 2
    assert any("passive block" in m for m in sent_msgs)
    assert any("active block" in m for m in sent_msgs)
    # Only the notify+wake sub woke the agent, exactly once.
    trigger_mock.assert_awaited_once()
    assert "active block" in trigger_mock.await_args.args[2]


@pytest.mark.asyncio
async def test_notifier_wake_forwards_persisted_chat_type_and_user_id(kanban_home):
    """The active-wake call must carry the subscription's persisted chat_type and
    user_id so _trigger_gateway_agent's build_session_key resolves the
    operator's real (e.g. group) session instead of a hardcoded one."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="group wake", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="grp1",
            user_id="op-42", chat_type="group", delivery_mode="wake",
        )
        kb.block_task(conn, tid, reason="group block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    # Legacy/unstamped subs are only visible to the dispatch-owning
    # gateway (include_unowned); mimic that ownership like the
    # upstream notifier tests do.
    runner._kanban_dispatcher_lock_handle = object()
    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    trigger_mock = AsyncMock(return_value={"triggered_agent": True})
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("tools.send_message_tool._trigger_gateway_agent", new=trigger_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    trigger_mock.assert_awaited_once()
    kwargs = trigger_mock.await_args.kwargs
    assert kwargs.get("chat_type") == "group"
    assert kwargs.get("user_id") == "op-42"


@pytest.mark.asyncio
async def test_notifier_wake_only_skips_send_and_advances_cursor(kanban_home):
    """wake-only: NO passive send, the agent is woken exactly once, and the
    cursor advances so repeated ticks do not re-wake."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="wake only task", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="wake",
        )
        kb.block_task(conn, tid, reason="wake only block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    # Legacy/unstamped subs are only visible to the dispatch-owning
    # gateway (include_unowned); mimic that ownership like the
    # upstream notifier tests do.
    runner._kanban_dispatcher_lock_handle = object()

    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    trigger_mock = AsyncMock(return_value={"triggered_agent": True})
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("tools.send_message_tool._trigger_gateway_agent", new=trigger_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # wake-only never uses the passive transport...
    fake_adapter.send.assert_not_awaited()
    # ...and wakes the agent exactly once across several ticks (proves the
    # cursor advanced; otherwise it would re-wake on every poll).
    trigger_mock.assert_awaited_once()
    assert "wake only block" in trigger_mock.await_args.args[2]

    # The subscription survives (blocked is non-terminal) but its cursor moved
    # past the blocked event.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert int(subs[0]["last_event_id"]) > 0


# ---------------------------------------------------------------------------
# Regression: gateway watchers must not double-init the kanban DB.
#
# Both the notifier watcher (`_kanban_notifier_watcher`) and the dispatcher
# tick (`_tick_once_for_board`) used to call `_kb.connect(board=slug)`
# immediately followed by `_kb.init_db(board=slug)`. Since `connect()`
# already runs the schema + idempotent migration on first open per process,
# the explicit `init_db()` was redundant — and worse, `init_db()`
# deliberately busts the per-process cache and re-runs the migration on a
# *second* connection, which races the first.  On legacy DBs this surfaced
# as `duplicate column name: <col>` (now tolerated by
# `_add_column_if_missing`) and intermittent `database is locked` errors
# (issue #21378).
#
# The fix removes the `init_db()` calls in both watchers; this regression
# test pins that behaviour so we don't reintroduce them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_create_autosubscribes_on_explicit_board(kanban_home):
    """`/kanban --board <slug> create ...` must subscribe on that board.

    The gateway handler currently auto-subscribes after `/kanban create`,
    but the create detection must still work when the shared `--board`
    flag appears before the subcommand, and the subscription must land in
    that board's DB rather than the ambient/default board.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    kb.create_board("projx")

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type="dm",
        thread_id="20197",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice',
        source=source,
        message_id="462",
        reply_to_message_id=None,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)

    assert "subscribed" in out.lower()

    conn = kb.connect(board="projx")
    try:
        subs = kb.list_notify_subs(conn)
        tasks = kb.list_tasks(conn)
    finally:
        conn.close()

    assert [t.title for t in tasks] == ["hello"]
    assert len(subs) == 1
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "20197"
    assert subs[0]["delivery_metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_notifier_artifact_delivery_skips_missing_files(kanban_home, tmp_path, monkeypatch):
    """Missing artifact paths are silently skipped — they may have been
    referenced by name only. The notifier must not crash and must still
    deliver any artifacts that do exist."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    # Allow ``tmp_path`` through the media-delivery safety filter. See the
    # companion test for the full explanation.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
    finally:
        conn.close()

    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    try:
        kt._handle_complete({
            "summary": "one real, one ghost",
            "artifacts": [str(real_pdf), "/tmp/definitely-does-not-exist.pdf"],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()

    fake_adapter = MagicMock()
    fake_adapter.name = "telegram"

    documents_uploaded: list = []

    async def _send(chat_id, msg, metadata=None):
        runner._running = False

    async def _send_document(chat_id, file_path, metadata=None, **_kw):
        documents_uploaded.append(file_path)

    fake_adapter.send = AsyncMock(side_effect=_send)
    fake_adapter.send_document = AsyncMock(side_effect=_send_document)
    fake_adapter.send_multiple_images = AsyncMock()
    from gateway.platforms.base import BasePlatformAdapter
    fake_adapter.extract_local_files = BasePlatformAdapter.extract_local_files

    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Only the real file was uploaded.
    assert len(documents_uploaded) == 1
    assert "real.pdf" in documents_uploaded[0]
