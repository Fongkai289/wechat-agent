"""Saved Q&A conversations in SQLite.

The store is addressed by its historical JSON path (``qa_conversations.json``)
because checkpoint locations are derived from it. Rows live next to it in
``<stem>.sqlite3``; a legacy JSON file is imported once and left untouched as a
backup. Each message is one row, so updating an answer in progress rewrites a
single row instead of the whole history.
"""

from __future__ import annotations

import functools
import os
import json
import re
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .qa_answers import validate_qa_answer
from .qa_recovery import discard_checkpoint as discard_qa_checkpoint, recovery_info as qa_recovery_info

MAX_STORED_MESSAGES = 200
DEFAULT_TITLE = "新对话"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversations(
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages(
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    turn_id TEXT,
    pending INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL,
    PRIMARY KEY(conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_qa_messages_turn ON messages(conversation_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_qa_messages_pending ON messages(pending) WHERE pending=1;
CREATE INDEX IF NOT EXISTS idx_qa_conversations_updated ON conversations(updated_at DESC, position);
"""

_qa_write_lock = threading.RLock()
_initialized: set[str] = set()
_init_lock = threading.Lock()


def database_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".sqlite3")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_qa_store(path: Path) -> dict[str, Any]:
    """Read a legacy JSON store; used only for the one-time import."""
    path = Path(path)
    if not path.exists():
        return {"conversations": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"conversations": []}
    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list):
        return {"conversations": []}
    return data


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    db_path = database_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(db_path.resolve())
    if not db_path.exists():
        _initialized.discard(key)
        # Private chat history: create owner-only. SQLite gives its -wal and
        # -shm files the same permissions as the database file.
        os.close(os.open(db_path, os.O_CREAT | os.O_WRONLY, 0o600))
    with closing(sqlite3.connect(db_path, timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if key not in _initialized:
            with _init_lock:
                if key not in _initialized:
                    initialize(conn, Path(path))
                    _initialized.add(key)
        yield conn


def initialize(conn: sqlite3.Connection, legacy_json: Path) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    with conn:
        imported = conn.execute("SELECT value FROM meta WHERE key='legacy_json_imported'").fetchone()
        if imported:
            return
        has_rows = conn.execute("SELECT 1 FROM conversations LIMIT 1").fetchone()
        if not has_rows:
            for position, conv in enumerate(load_qa_store(legacy_json)["conversations"]):
                if not isinstance(conv, dict) or not conv.get("id"):
                    continue
                messages = conv.get("messages") if isinstance(conv.get("messages"), list) else []
                insert_conversation(conn, {
                    "id": str(conv["id"]),
                    "title": str(conv.get("title") or DEFAULT_TITLE),
                    "created_at": str(conv.get("created_at") or ""),
                    "updated_at": str(conv.get("updated_at") or ""),
                }, position)
                write_messages(conn, str(conv["id"]), [m for m in messages if isinstance(m, dict)])
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('legacy_json_imported', ?)", (now_text(),))


def insert_conversation(conn: sqlite3.Connection, conv: dict[str, Any], position: int | None = None) -> None:
    if position is None:
        position = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM conversations").fetchone()[0]
    conn.execute(
        "INSERT INTO conversations(id, title, created_at, updated_at, position) VALUES (?,?,?,?,?)",
        (conv["id"], conv["title"], conv["created_at"], conv["updated_at"], position),
    )


def write_messages(conn: sqlite3.Connection, conversation_id: str, messages: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
    conn.executemany(
        "INSERT INTO messages(conversation_id, seq, turn_id, pending, data) VALUES (?,?,?,?,?)",
        [
            (conversation_id, seq, message.get("turn_id"), 1 if message.get("pending") else 0,
             json.dumps(message, ensure_ascii=False))
            for seq, message in enumerate(messages)
        ],
    )


def read_messages(conn: sqlite3.Connection, conversation_id: str) -> list[dict[str, Any]]:
    return [json.loads(row["data"]) for row in conn.execute(
        "SELECT data FROM messages WHERE conversation_id=? ORDER BY seq", (conversation_id,))]


def conversation_row(conn: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?", (conversation_id,)
    ).fetchone()


def conversation_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": read_messages(conn, row["id"]),
    }


def serialized_qa_write(function):
    @functools.wraps(function)
    def locked(*args, **kwargs):
        with _qa_write_lock:
            return function(*args, **kwargs)
    return locked


def list_qa_conversations(path: Path) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count
               FROM conversations c ORDER BY c.updated_at DESC, c.position ASC"""
        ).fetchall()
    return [
        {
            "id": row["id"] or "",
            "title": row["title"] or DEFAULT_TITLE,
            "created_at": row["created_at"] or "",
            "updated_at": row["updated_at"] or "",
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]


def get_qa_conversation(path: Path, conversation_id: str) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conversation_row(conn, conversation_id)
        if row is None:
            return None
        conv = conversation_dict(conn, row)
    for message in conv["messages"]:
        if message.get("role") == "assistant":
            message.update(qa_recovery_info(path, message))
    return conv


@serialized_qa_write
def update_qa_turn(path, conversation_id, turn_id, message):
    now = now_text()
    with connect(path) as conn, conn:
        target = conn.execute(
            "SELECT seq FROM messages WHERE conversation_id=? AND turn_id=? ORDER BY seq LIMIT 1",
            (conversation_id, turn_id),
        ).fetchone()
        if target is None:
            raise ValueError("原回答已不存在，未覆盖其他对话")
        clean = sanitize_qa_messages_for_store([message])[0]
        conn.execute(
            "UPDATE messages SET turn_id=?, pending=?, data=? WHERE conversation_id=? AND seq=?",
            (clean.get("turn_id"), 1 if clean.get("pending") else 0,
             json.dumps(clean, ensure_ascii=False), conversation_id, target["seq"]),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        return conversation_dict(conn, conversation_row(conn, conversation_id))


@serialized_qa_write
def interrupt_pending_qa(path: Path) -> None:
    retained = set()
    with connect(path) as conn, conn:
        for row in conn.execute("SELECT conversation_id, seq, data FROM messages").fetchall():
            message = json.loads(row["data"])
            if message.pop("pending", False):
                message.update(content="服务已重启，上次回答中断，未自动重试", error="服务已重启")
                conn.execute(
                    "UPDATE messages SET pending=0, data=? WHERE conversation_id=? AND seq=?",
                    (json.dumps(message, ensure_ascii=False), row["conversation_id"], row["seq"]),
                )
            if message.get("error") and not message.get("stopped"):
                retained.add(message.get("turn_id"))
    for candidate in (Path(path).parent / "qa_checkpoints").glob("*.sqlite3"):
        if candidate.stem not in retained:
            discard_qa_checkpoint(path, candidate.stem)


@serialized_qa_write
def create_qa_conversation(path: Path, title: str = "") -> dict[str, Any]:
    now = now_text()
    conv = {
        "id": uuid.uuid4().hex,
        "title": title.strip() or DEFAULT_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with connect(path) as conn, conn:
        insert_conversation(conn, conv)
    return conv


@serialized_qa_write
def save_qa_conversation(path: Path, conversation_id: str, messages: list[dict[str, Any]], title: str = "") -> dict[str, Any]:
    now = now_text()
    clean_messages = sanitize_qa_messages_for_store(messages)
    resolved_title = title.strip() or title_from_qa_messages(clean_messages) or DEFAULT_TITLE
    with connect(path) as conn, conn:
        row = conversation_row(conn, conversation_id) if conversation_id else None
        if row is None:
            found = {"id": conversation_id or uuid.uuid4().hex, "created_at": now, "title": "",
                     "updated_at": now}
            insert_conversation(conn, found)
            old_turns: set[Any] = set()
        else:
            found = {"id": row["id"], "created_at": row["created_at"], "title": row["title"],
                     "updated_at": row["updated_at"]}
            old_turns = {m.get("turn_id") for m in read_messages(conn, row["id"])}
        found["title"] = title.strip() or (found["title"] if found["title"] != DEFAULT_TITLE else "") or resolved_title
        found["updated_at"] = now
        found["messages"] = clean_messages[-MAX_STORED_MESSAGES:]
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                     (found["title"], now, found["id"]))
        write_messages(conn, found["id"], found["messages"])
    for turn_id in old_turns - {m.get("turn_id") for m in found["messages"]}:
        discard_qa_checkpoint(path, turn_id)
    return found


@serialized_qa_write
def rename_qa_conversation(path: Path, conversation_id: str, title: str) -> dict[str, Any] | None:
    now = now_text()
    with connect(path) as conn, conn:
        cursor = conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                              (title.strip()[:80] or DEFAULT_TITLE, now, conversation_id))
        if not cursor.rowcount:
            return None
        return conversation_dict(conn, conversation_row(conn, conversation_id))


@serialized_qa_write
def delete_qa_conversation(path: Path, conversation_id: str) -> bool:
    with connect(path) as conn, conn:
        turns = [row[0] for row in conn.execute(
            "SELECT turn_id FROM messages WHERE conversation_id=?", (conversation_id,))]
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        deleted = conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,)).rowcount
    if not deleted:
        return False
    for turn_id in turns:
        discard_qa_checkpoint(path, turn_id)
    return True


def sanitize_qa_messages_for_store(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        row: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            if re.fullmatch(r'[0-9a-f]{32}', str(item.get('turn_id') or '')):
                row['turn_id'] = item['turn_id']
            if item.get("error"):
                row["error"] = str(item["error"])
            if item.get("pending"):
                row["pending"] = True
            sources = item.get("sources") if isinstance(item.get("sources"), list) else []
            row["sources"] = sources
            if item.get("answer_data"):
                try:
                    row["answer_data"] = validate_qa_answer(item["answer_data"], len(sources))
                except ValueError:
                    pass  # Legacy or malformed records remain readable as plain text.
            row["context_count"] = int(item.get("context_count") or 0)
            if item.get("processing_ms") is not None:
                row["processing_ms"] = int(item.get("processing_ms") or 0)
            elif item.get("elapsed_ms") is not None:
                row["processing_ms"] = int(item.get("elapsed_ms") or 0)
            if item.get("stopped"):
                row["stopped"] = True
            retrieval = item.get("retrieval") if isinstance(item.get("retrieval"), dict) else {}
            if retrieval:
                row["retrieval"] = {
                    key: retrieval.get(key)
                    for key in ("mode", "scope", "time_scope", "candidate_count", "scored_count", "context_count", "matched_people", "related_people",
                                "query_plan", "conversation_state", "trace", "coverage", "usage", "answer_reviews", "quality_status",
                                "current_task", "active_step", "model_metrics", "answer_timings", "transport_calls")
                    if key in retrieval
                }
        out.append(row)
    return out


def title_from_qa_messages(messages: list[dict[str, Any]]) -> str:
    for item in messages:
        if item.get("role") == "user":
            text = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
            return text[:28] + ("..." if len(text) > 28 else "")
    return ""
