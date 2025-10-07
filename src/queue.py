"""Persistent queue storage for gallery work items."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_QUEUE_PATH = Path("db") / "que.db"


@dataclass(slots=True)
class QueueRow:
    """Materialised representation of a row in the persistent queue."""

    id: int
    gallery_id: str
    post_id: str
    action: str
    title: Optional[str]
    manual: bool
    delay: float
    available_at: float
    payload: Optional[Dict[str, Any]]
    state: str


class QueueDatabase:
    """SQLite-backed queue that survives application restarts."""

    def __init__(self, path: Path | str = DEFAULT_QUEUE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._initialise()

    def _initialise(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gallery_id TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    title TEXT,
                    manual INTEGER NOT NULL DEFAULT 0,
                    delay REAL NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    payload TEXT,
                    state TEXT NOT NULL DEFAULT 'queued',
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queue_gallery_state
                ON queue_items(gallery_id, state, available_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queue_gallery_action
                ON queue_items(gallery_id, action)
                """
            )

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)

    async def enqueue(
        self,
        gallery_id: str,
        post_id: str,
        action: str,
        *,
        title: Optional[str] = None,
        manual: bool = False,
        delay: float = 0.0,
        available_at: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        front: bool = False,
    ) -> QueueRow:
        now = time.time()
        if available_at is None:
            available_at = now + max(delay, 0.0)
        if front:
            available_at = min(available_at, now) - 0.001

        record = await self._execute_insert(
            gallery_id,
            post_id,
            action,
            title,
            manual,
            delay,
            available_at,
            payload,
            now,
        )
        return record

    async def _execute_insert(
        self,
        gallery_id: str,
        post_id: str,
        action: str,
        title: Optional[str],
        manual: bool,
        delay: float,
        available_at: float,
        payload: Optional[Dict[str, Any]],
        now: float,
    ) -> QueueRow:
        payload_json = json.dumps(payload, ensure_ascii=False) if payload else None

        async with self._lock:
            row_id = await asyncio.to_thread(
                self._insert_row,
                gallery_id,
                post_id,
                action,
                title,
                manual,
                delay,
                available_at,
                payload_json,
                now,
            )
            return await asyncio.to_thread(self._fetch_row, row_id)

    def _insert_row(
        self,
        gallery_id: str,
        post_id: str,
        action: str,
        title: Optional[str],
        manual: bool,
        delay: float,
        available_at: float,
        payload_json: Optional[str],
        now: float,
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO queue_items (
                    gallery_id, post_id, action, title, manual, delay,
                    available_at, payload, state, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    gallery_id,
                    post_id,
                    action,
                    title,
                    int(manual),
                    delay,
                    available_at,
                    payload_json,
                    now,
                    now,
                ),
            )
        return int(cursor.lastrowid)

    def _fetch_row(self, row_id: int) -> QueueRow:
        cursor = self._conn.execute(
            "SELECT * FROM queue_items WHERE id = ?",
            (row_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Queue row {row_id} not found")
        return self._row_from_result(row)

    def _row_from_result(self, row: sqlite3.Row) -> QueueRow:
        payload = row["payload"]
        decoded: Optional[Dict[str, Any]] = None
        if payload:
            decoded = json.loads(payload)
        return QueueRow(
            id=int(row["id"]),
            gallery_id=str(row["gallery_id"]),
            post_id=str(row["post_id"]),
            action=str(row["action"]),
            title=row["title"],
            manual=bool(row["manual"]),
            delay=float(row["delay"] or 0.0),
            available_at=float(row["available_at"]),
            payload=decoded,
            state=str(row["state"]),
        )

    async def acquire_next(
        self, gallery_id: str
    ) -> tuple[Optional[QueueRow], Optional[float]]:
        now = time.time()
        async with self._lock:
            return await asyncio.to_thread(self._acquire_next_row, gallery_id, now)

    def _acquire_next_row(
        self, gallery_id: str, now: float
    ) -> tuple[Optional[QueueRow], Optional[float]]:
        while True:
            cursor = self._conn.execute(
                """
                SELECT * FROM queue_items
                WHERE gallery_id = ? AND state = 'queued'
                ORDER BY available_at ASC, id ASC
                LIMIT 1
                """,
                (gallery_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None, None
            available_at = float(row["available_at"])
            if available_at > now:
                return None, available_at
            with self._conn:
                updated = self._conn.execute(
                    """
                    UPDATE queue_items
                    SET state = 'processing', updated_ts = ?
                    WHERE id = ? AND state = 'queued'
                    """,
                    (now, row["id"]),
                )
            if updated.rowcount:
                return self._row_from_result(row), None

    async def reset_processing(self, gallery_id: str) -> None:
        now = time.time()
        async with self._lock:
            await asyncio.to_thread(self._reset_processing_rows, gallery_id, now)

    def _reset_processing_rows(self, gallery_id: str, now: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE queue_items
                SET state = 'queued', updated_ts = ?
                WHERE gallery_id = ? AND state = 'processing'
                """,
                (now, gallery_id),
            )

    async def reschedule(
        self, queue_id: int, *, delay: float = 0.0, front: bool = False
    ) -> None:
        now = time.time()
        available_at = now + max(delay, 0.0)
        if front:
            available_at = now - 0.001
        async with self._lock:
            await asyncio.to_thread(
                self._reschedule_row, queue_id, available_at, delay, now
            )

    def _reschedule_row(
        self, queue_id: int, available_at: float, delay: float, now: float
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE queue_items
                SET available_at = ?, delay = ?, state = 'queued', updated_ts = ?
                WHERE id = ?
                """,
                (available_at, delay, now, queue_id),
            )

    async def complete(self, queue_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_row, queue_id)

    def _delete_row(self, queue_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM queue_items WHERE id = ?",
                (queue_id,),
            )

    async def delete_for_post(self, gallery_id: str, post_id: str) -> None:
        """Remove all queue entries associated with a specific post."""

        async with self._lock:
            await asyncio.to_thread(
                self._delete_rows_for_post,
                gallery_id,
                post_id,
            )

    def _delete_rows_for_post(self, gallery_id: str, post_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM queue_items WHERE gallery_id = ? AND post_id = ?",
                (gallery_id, post_id),
            )

    async def count_due(self, gallery_id: str) -> int:
        now = time.time()
        async with self._lock:
            return await asyncio.to_thread(self._count_due_rows, gallery_id, now)

    def _count_due_rows(self, gallery_id: str, now: float) -> int:
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM queue_items
            WHERE gallery_id = ? AND state = 'queued' AND available_at <= ?
            """,
            (gallery_id, now),
        )
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    async def list_due(self, gallery_id: str, limit: int) -> list[QueueRow]:
        now = time.time()
        async with self._lock:
            rows = await asyncio.to_thread(
                self._list_due_rows, gallery_id, now, limit
            )
        return rows

    def _list_due_rows(
        self, gallery_id: str, now: float, limit: int
    ) -> list[QueueRow]:
        cursor = self._conn.execute(
            """
            SELECT * FROM queue_items
            WHERE gallery_id = ? AND state = 'queued' AND available_at <= ?
            ORDER BY available_at ASC, id ASC
            LIMIT ?
            """,
            (gallery_id, now, limit),
        )
        return [self._row_from_result(row) for row in cursor.fetchall()]

    async def count_global_due(self) -> int:
        now = time.time()
        async with self._lock:
            return await asyncio.to_thread(self._count_global_due_rows, now)

    def _count_global_due_rows(self, now: float) -> int:
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM queue_items
            WHERE available_at <= ? AND state IN ('queued', 'processing')
            """,
            (now,),
        )
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    async def list_global_due(self, limit: int) -> list[QueueRow]:
        now = time.time()
        async with self._lock:
            rows = await asyncio.to_thread(
                self._list_global_due_rows, now, limit
            )
        return rows

    def _list_global_due_rows(self, now: float, limit: int) -> list[QueueRow]:
        cursor = self._conn.execute(
            """
            SELECT * FROM queue_items
            WHERE available_at <= ? AND state IN ('queued', 'processing')
            ORDER BY CASE WHEN state = 'processing' THEN 0 ELSE 1 END,
                     available_at ASC, id ASC
            LIMIT ?
            """,
            (now, limit),
        )
        return [self._row_from_result(row) for row in cursor.fetchall()]

    async def count_global_scheduled(self) -> int:
        now = time.time()
        async with self._lock:
            return await asyncio.to_thread(self._count_global_scheduled_rows, now)

    def _count_global_scheduled_rows(self, now: float) -> int:
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM queue_items
            WHERE available_at > ? AND state = 'queued'
            """,
            (now,),
        )
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    async def list_global_scheduled(self, limit: int) -> list[QueueRow]:
        now = time.time()
        async with self._lock:
            rows = await asyncio.to_thread(
                self._list_global_scheduled_rows, now, limit
            )
        return rows

    def _list_global_scheduled_rows(self, now: float, limit: int) -> list[QueueRow]:
        cursor = self._conn.execute(
            """
            SELECT * FROM queue_items
            WHERE available_at > ? AND state = 'queued'
            ORDER BY available_at ASC, id ASC
            LIMIT ?
            """,
            (now, limit),
        )
        return [self._row_from_result(row) for row in cursor.fetchall()]

    async def pending_post_ids(self, gallery_id: str, action: str) -> set[str]:
        async with self._lock:
            return await asyncio.to_thread(
                self._pending_post_ids, gallery_id, action
            )

    def _pending_post_ids(self, gallery_id: str, action: str) -> set[str]:
        cursor = self._conn.execute(
            """
            SELECT DISTINCT post_id FROM queue_items
            WHERE gallery_id = ? AND action = ?
            """,
            (gallery_id, action),
        )
        return {str(row[0]) for row in cursor.fetchall()}

    async def titles_for_gallery(self, gallery_id: str) -> dict[str, str]:
        async with self._lock:
            return await asyncio.to_thread(self._titles_for_gallery, gallery_id)

    def _titles_for_gallery(self, gallery_id: str) -> dict[str, str]:
        cursor = self._conn.execute(
            """
            SELECT post_id, title FROM queue_items
            WHERE gallery_id = ? AND title IS NOT NULL AND title <> ''
            """,
            (gallery_id,),
        )
        titles: dict[str, str] = {}
        for row in cursor.fetchall():
            titles[str(row[0])] = str(row[1])
        return titles

    async def load_followups(self, gallery_id: str) -> dict[str, set[int]]:
        async with self._lock:
            return await asyncio.to_thread(self._load_followups, gallery_id)

    def _load_followups(self, gallery_id: str) -> dict[str, set[int]]:
        cursor = self._conn.execute(
            """
            SELECT id, post_id FROM queue_items
            WHERE gallery_id = ? AND action IN ('followup', 'comment_followup')
                AND manual = 0
            """,
            (gallery_id,),
        )
        results: dict[str, set[int]] = {}
        for row in cursor.fetchall():
            results.setdefault(str(row["post_id"]), set()).add(int(row["id"]))
        return results

    async def comment_followup_counts(self, gallery_id: str) -> dict[str, int]:
        async with self._lock:
            return await asyncio.to_thread(
                self._comment_followup_counts, gallery_id
            )

    def _comment_followup_counts(self, gallery_id: str) -> dict[str, int]:
        cursor = self._conn.execute(
            """
            SELECT post_id, COUNT(*) AS cnt
            FROM queue_items
            WHERE gallery_id = ? AND action = 'comment_followup'
            GROUP BY post_id
            """,
            (gallery_id,),
        )
        counts: dict[str, int] = {}
        for row in cursor.fetchall():
            counts[str(row["post_id"])] = int(row["cnt"])
        return counts

