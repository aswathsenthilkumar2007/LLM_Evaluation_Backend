"""SQLite persistence for conversations and their evaluated turns."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.models.schemas import CompareResponse

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "conversations.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_locked INTEGER NOT NULL DEFAULT 0 CHECK (title_locked IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    reference_answer TEXT,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_turns_order
    ON conversation_turns(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _database_path() -> Path:
    configured_path = os.getenv("CONVERSATION_DB_PATH", "").strip()
    path = Path(configured_path) if configured_path else DEFAULT_DATABASE_PATH
    return path.expanduser().resolve()


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _conversation_title(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    if not normalized:
        return "New chat"
    if len(normalized) <= 72:
        return normalized
    return normalized[:69].rstrip() + "..."


def create_conversation(title: str | None = None) -> dict[str, Any]:
    now = _now()
    conversation_id = str(uuid4())
    resolved_title = title or "New chat"
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, title_locked, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, resolved_title, int(title is not None), now, now),
        )
    return {
        "id": conversation_id,
        "title": resolved_title,
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
    }


def list_conversations() -> list[dict[str, Any]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(t.id) AS turn_count
            FROM conversations AS c
            LEFT JOIN conversation_turns AS t ON t.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with _connection() as connection:
        conversation = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(t.id) AS turn_count
            FROM conversations AS c
            LEFT JOIN conversation_turns AS t ON t.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None

        turns = connection.execute(
            """
            SELECT id, prompt, reference_answer, result_json, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

    return {
        **dict(conversation),
        "turns": [
            {
                "id": turn["id"],
                "prompt": turn["prompt"],
                "reference_answer": turn["reference_answer"],
                "created_at": turn["created_at"],
                "result": json.loads(turn["result_json"]),
            }
            for turn in turns
        ],
    }


def rename_conversation(conversation_id: str, title: str) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, title_locked = 1, updated_at = ?
            WHERE id = ?
            """,
            (title, _now(), conversation_id),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


def add_turn(
    conversation_id: str,
    prompt: str,
    reference_answer: str | None,
    result: CompareResponse,
) -> dict[str, Any] | None:
    now = _now()
    serialized_result = result.model_dump_json()
    with _connection() as connection:
        conversation = connection.execute(
            "SELECT title, title_locked FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None

        turn_count = connection.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        if turn_count == 0 and not conversation["title_locked"]:
            title = _conversation_title(prompt)
            connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

        cursor = connection.execute(
            """
            INSERT INTO conversation_turns
                (conversation_id, prompt, reference_answer, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                prompt,
                reference_answer,
                serialized_result,
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        turn_id = cursor.lastrowid

    return {
        "id": turn_id,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "created_at": now,
        "result": json.loads(serialized_result),
    }
