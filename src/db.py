"""SQLite helpers for the DCInside archiver."""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .dcinside import Comment, PostDetail


DEFAULT_DB_DIR = Path("db")


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


MODIFICATION_MARKER = "(수정됨 - "


def _strip_history(text: Optional[str]) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        if line.startswith(MODIFICATION_MARKER):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_history(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [line for line in text.splitlines() if line.startswith(MODIFICATION_MARKER)]


def _find_removed_chunks(previous: str, current: str) -> list[str]:
    if not previous or previous == current:
        return []
    removed: list[str] = []
    chunk: list[str] = []
    for line in difflib.ndiff(previous.splitlines(), current.splitlines()):
        if line.startswith("- "):
            chunk.append(line[2:])
            continue
        if chunk and not line.startswith("? "):
            merged = "\n".join(chunk).strip()
            if merged:
                removed.append(merged)
            chunk = []
    if chunk:
        merged = "\n".join(chunk).strip()
        if merged:
            removed.append(merged)
    return removed


def _format_removed_chunk(chunk: str) -> Optional[str]:
    cleaned = chunk.strip()
    if not cleaned:
        return None
    return f"{MODIFICATION_MARKER}{cleaned})"


def _merge_content_with_history(current: str, *, previous_history: Sequence[str], removed_chunks: Sequence[str]) -> str:
    history = list(previous_history)
    for chunk in removed_chunks:
        for line in chunk.split("\n"):
            formatted = _format_removed_chunk(line)
            if formatted and formatted not in history:
                history.append(formatted)
    if not history:
        return current
    main = current.rstrip("\n")
    if main:
        return f"{main}\n" + "\n".join(history)
    return "\n".join(history)


@dataclass(slots=True)
class StoredComment:
    comment_id: str
    post_id: str
    parent_id: Optional[str]
    nickname: str
    user_id: Optional[str]
    ip: Optional[str]
    created_at: str
    content: str
    dccons: Sequence[str] = field(default_factory=tuple)
    is_deleted: bool = False


class GalleryDatabase:
    """Lightweight SQLite wrapper dedicated to a single gallery."""

    def __init__(self, gallery_id: str, *, root: Path = DEFAULT_DB_DIR) -> None:
        self.gallery_id = gallery_id
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{gallery_id}.db"
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._initialise()

    def _initialise(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    gallery_id TEXT NOT NULL,
                    title TEXT,
                    nickname TEXT,
                    user_id TEXT,
                    ip TEXT,
                    created_at TEXT,
                    content TEXT,
                    content_raw TEXT,
                    view_count INTEGER,
                    recommend_count INTEGER,
                    dislike_count INTEGER,
                    is_deleted INTEGER DEFAULT 0,
                    images TEXT,
                    created_ts TEXT NOT NULL,
                    updated_ts TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS post_images (
                    post_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    url TEXT,
                    content_type TEXT,
                    data BLOB,
                    PRIMARY KEY (post_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT,
                    post_id TEXT,
                    parent_id TEXT,
                    nickname TEXT,
                    user_id TEXT,
                    ip TEXT,
                    created_at TEXT,
                    content TEXT,
                    content_raw TEXT,
                    dccons TEXT,
                    is_deleted INTEGER DEFAULT 0,
                    PRIMARY KEY (comment_id, post_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posts_gallery_created
                ON posts(gallery_id, created_at)
                """
            )
            self._ensure_comment_dccon_column()
            self._ensure_post_content_raw_column()
            self._ensure_comment_content_raw_column()

    def _ensure_comment_dccon_column(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(comments)")
        columns = {row[1] for row in cur.fetchall()}
        if "dccons" in columns:
            return
        if "images" in columns:
            self._conn.execute("ALTER TABLE comments RENAME COLUMN images TO dccons")
        else:
            self._conn.execute("ALTER TABLE comments ADD COLUMN dccons TEXT")

    def _ensure_post_content_raw_column(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(posts)")
        columns = {row[1] for row in cur.fetchall()}
        if "content_raw" not in columns:
            self._conn.execute("ALTER TABLE posts ADD COLUMN content_raw TEXT")

    def _ensure_comment_content_raw_column(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(comments)")
        columns = {row[1] for row in cur.fetchall()}
        if "content_raw" not in columns:
            self._conn.execute("ALTER TABLE comments ADD COLUMN content_raw TEXT")

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    async def has_post(self, post_id: str) -> bool:
        def _query() -> bool:
            cur = self._conn.execute(
                "SELECT 1 FROM posts WHERE post_id = ? LIMIT 1", (post_id,)
            )
            return cur.fetchone() is not None

        return await asyncio.to_thread(_query)

    async def known_post_ids(self) -> set[str]:
        def _query() -> set[str]:
            cur = self._conn.execute("SELECT post_id FROM posts")
            return {row[0] for row in cur.fetchall()}

        return await asyncio.to_thread(_query)

    def file_size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    async def upsert_post(
        self,
        post: PostDetail,
        *,
        images: Optional[Sequence[tuple[str, bytes, Optional[str]]]] = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._upsert_post_sync, post, images)

    def _upsert_post_sync(
        self, post: PostDetail, images: Optional[Sequence[tuple[str, bytes, Optional[str]]]]
    ) -> None:
        existing_row = self._conn.execute(
            "SELECT content, content_raw FROM posts WHERE post_id = ?",
            (post.post_id,),
        ).fetchone()
        existing_content = ((existing_row["content"] if existing_row else "") or "")
        existing_history = _extract_history(existing_content)
        previous_raw = ""
        if existing_row:
            stored_raw = existing_row["content_raw"]
            previous_raw = stored_raw if stored_raw is not None else _strip_history(existing_content)
        new_content_raw = post.content or ""
        removed_segments = _find_removed_chunks(previous_raw, new_content_raw)
        stored_content = _merge_content_with_history(
            new_content_raw,
            previous_history=existing_history,
            removed_chunks=removed_segments,
        )

        existing_comment_rows = self._conn.execute(
            """
            SELECT comment_id, content, content_raw, is_deleted
            FROM comments
            WHERE post_id = ?
            """,
            (post.post_id,),
        ).fetchall()

        existing_comments: dict[str, sqlite3.Row] = {}
        for row in existing_comment_rows:
            comment_id = (row["comment_id"] or "").strip()
            if not comment_id:
                continue
            existing_comments[comment_id] = row

        images_json = json.dumps(list(post.images), ensure_ascii=False)
        now = _utcnow()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO posts (
                    post_id, gallery_id, title, nickname, user_id, ip, created_at,
                    content, content_raw, view_count, recommend_count, dislike_count,
                    is_deleted, images, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    title = excluded.title,
                    nickname = excluded.nickname,
                    user_id = excluded.user_id,
                    ip = excluded.ip,
                    created_at = excluded.created_at,
                    content = excluded.content,
                    content_raw = excluded.content_raw,
                    view_count = excluded.view_count,
                    recommend_count = excluded.recommend_count,
                    dislike_count = excluded.dislike_count,
                    is_deleted = excluded.is_deleted,
                    images = excluded.images,
                    updated_ts = excluded.updated_ts
                """,
                (
                    post.post_id,
                    post.gallery_id,
                    post.title,
                    post.author.nickname,
                    post.author.user_id,
                    post.author.ip,
                    post.created_at,
                    stored_content,
                    new_content_raw,
                    post.view_count,
                    post.recommend_count,
                    post.dislike_count,
                    int(post.is_deleted),
                    images_json,
                    now,
                    now,
                ),
            )

            for comment in post.comments:
                dccon_json = json.dumps(list(comment.dccons), ensure_ascii=False)
                comment_id = (comment.comment_id or "").strip()
                if not comment_id:
                    continue
                existing_comment = existing_comments.pop(comment_id, None)
                existing_comment_content = (
                    (existing_comment["content"] if existing_comment else "") or ""
                )
                comment_history = _extract_history(existing_comment_content)
                previous_comment_raw = ""
                if existing_comment:
                    stored_raw = existing_comment["content_raw"]
                    previous_comment_raw = (
                        stored_raw
                        if stored_raw is not None
                        else _strip_history(existing_comment_content)
                    )
                new_comment_raw = comment.content or ""
                removed_comment_segments = _find_removed_chunks(
                    previous_comment_raw, new_comment_raw
                )
                stored_comment_content = _merge_content_with_history(
                    new_comment_raw,
                    previous_history=comment_history,
                    removed_chunks=removed_comment_segments,
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO comments (
                        comment_id, post_id, parent_id, nickname, user_id, ip,
                        created_at, content, content_raw, dccons, is_deleted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        comment_id,
                        post.post_id,
                        comment.parent_id,
                        comment.author.nickname,
                        comment.author.user_id,
                        comment.author.ip,
                        comment.created_at,
                        stored_comment_content,
                        new_comment_raw,
                        dccon_json,
                        int(comment.is_deleted),
                    ),
                )

            for comment_id, row in existing_comments.items():
                existing_content = (row["content"] or "") or ""
                history_lines = _extract_history(existing_content)
                previous_raw_content = row["content_raw"]
                if previous_raw_content is None:
                    previous_raw_content = _strip_history(existing_content)
                removed_segments = _find_removed_chunks(previous_raw_content or "", "")
                updated_content = _merge_content_with_history(
                    "",
                    previous_history=history_lines,
                    removed_chunks=removed_segments,
                )
                if not updated_content and existing_content:
                    updated_content = existing_content
                if (
                    updated_content == existing_content
                    and int(row["is_deleted"] or 0) == 1
                    and (row["content_raw"] or "") == ""
                ):
                    continue
                self._conn.execute(
                    """
                    UPDATE comments
                    SET content = ?, content_raw = ?, is_deleted = 1
                    WHERE post_id = ? AND comment_id = ?
                    """,
                    (
                        updated_content,
                        "",
                        post.post_id,
                        comment_id,
                    ),
                )

            if images is not None:
                self._conn.execute(
                    "DELETE FROM post_images WHERE post_id = ?",
                    (post.post_id,),
                )
                for index, (url, data, content_type) in enumerate(images):
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO post_images (
                            post_id, position, url, content_type, data
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            post.post_id,
                            index,
                            url,
                            content_type,
                            sqlite3.Binary(data),
                        ),
                    )

    async def update_counts(
        self,
        post_id: str,
        *,
        view_count: Optional[int] = None,
        recommend_count: Optional[int] = None,
        dislike_count: Optional[int] = None,
        is_deleted: Optional[bool] = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._update_counts_sync,
                post_id,
                view_count,
                recommend_count,
                dislike_count,
                is_deleted,
            )

    def _update_counts_sync(
        self,
        post_id: str,
        view_count: Optional[int],
        recommend_count: Optional[int],
        dislike_count: Optional[int],
        is_deleted: Optional[bool],
    ) -> None:
        fields = []
        params: list[object] = []
        if view_count is not None:
            fields.append("view_count = ?")
            params.append(view_count)
        if recommend_count is not None:
            fields.append("recommend_count = ?")
            params.append(recommend_count)
        if dislike_count is not None:
            fields.append("dislike_count = ?")
            params.append(dislike_count)
        if is_deleted is not None:
            fields.append("is_deleted = ?")
            params.append(int(is_deleted))
        fields.append("updated_ts = ?")
        params.append(_utcnow())
        params.append(post_id)

        if len(fields) == 1:  # Only updated_ts changed
            return

        with self._conn:
            self._conn.execute(
                f"UPDATE posts SET {', '.join(fields)} WHERE post_id = ?",
                params,
            )

    async def count_posts(self) -> int:
        def _query() -> int:
            cur = self._conn.execute("SELECT COUNT(*) FROM posts")
            result = cur.fetchone()
            return int(result[0]) if result else 0

        return await asyncio.to_thread(_query)

    def _paginate_posts_sync(
        self,
        where_clause: str = "",
        params: Sequence[object] = (),
        *,
        page: int,
        page_size: int,
        deleted_only: bool = False,
    ) -> Tuple[List[sqlite3.Row], int]:
        page = max(page, 1)
        offset = (page - 1) * page_size
        clauses: list[str] = []
        if where_clause:
            clauses.append(f"({where_clause})")
        if deleted_only:
            clauses.append("is_deleted = 1")
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        total_cur = self._conn.execute(
            f"SELECT COUNT(*) FROM posts{where_sql}", params
        )
        total_row = total_cur.fetchone()
        total = int(total_row[0]) if total_row else 0

        cur = self._conn.execute(
            f"""
            SELECT post_id, title, nickname, user_id, created_at, view_count,
                   recommend_count, dislike_count, is_deleted
            FROM posts
            {where_sql}
            ORDER BY datetime(created_at) DESC, post_id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset),
        )
        rows = cur.fetchall()
        return rows, total

    async def list_posts(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> Tuple[List[sqlite3.Row], int]:
        return await asyncio.to_thread(
            self._paginate_posts_sync,
            "",
            (),
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )

    async def search_posts_paginated(
        self,
        query: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> Tuple[List[sqlite3.Row], int]:
        like_term = f"%{query}%"
        return await asyncio.to_thread(
            self._paginate_posts_sync,
            "title LIKE ? OR content LIKE ?",
            (like_term, like_term),
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )

    async def search_posts_by_author(
        self,
        nickname: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> Tuple[List[sqlite3.Row], int]:
        like_term = f"%{nickname}%"
        return await asyncio.to_thread(
            self._paginate_posts_sync,
            "nickname LIKE ?",
            (like_term,),
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )

    async def search_posts_by_user_id(
        self,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> Tuple[List[sqlite3.Row], int]:
        like_term = f"%{user_id}%"
        return await asyncio.to_thread(
            self._paginate_posts_sync,
            "user_id LIKE ?",
            (like_term,),
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )

    async def search_posts(self, query: str, *, limit: int = 25) -> List[sqlite3.Row]:
        like_term = f"%{query}%"

        def _query() -> List[sqlite3.Row]:
            cur = self._conn.execute(
                """
                SELECT post_id, title, nickname, created_at, view_count, recommend_count
                FROM posts
                WHERE title LIKE ? OR content LIKE ? OR nickname LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like_term, like_term, like_term, limit),
            )
            return cur.fetchall()

        return await asyncio.to_thread(_query)

    async def fetch_posts_for_summary(
        self,
        *,
        limit: int = 50,
        search: Optional[str] = None,
        author: Optional[str] = None,
        user_id: Optional[str] = None,
        deleted_only: bool = False,
    ) -> List[dict]:
        limit = max(1, min(int(limit or 1), 200))

        def _query() -> List[dict]:
            clauses: list[str] = []
            params: list[object] = []
            if search:
                like_term = f"%{search}%"
                clauses.append("(title LIKE ? OR content LIKE ?)")
                params.extend([like_term, like_term])
            if author:
                author_term = f"%{author}%"
                clauses.append("nickname LIKE ?")
                params.append(author_term)
            if user_id:
                user_term = f"%{user_id}%"
                clauses.append("user_id LIKE ?")
                params.append(user_term)
            if deleted_only:
                clauses.append("is_deleted = 1")
            where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""

            cur = self._conn.execute(
                f"""
                SELECT post_id, title, nickname, user_id, created_at, content,
                       view_count, recommend_count, dislike_count, is_deleted
                FROM posts
                {where_sql}
                ORDER BY COALESCE(datetime(created_at), created_at) DESC, post_id DESC
                LIMIT ?
                """,
                (*params, limit),
            )
            return [dict(row) for row in cur.fetchall()]

        return await asyncio.to_thread(_query)

    async def fetch_post(self, post_id: str) -> Optional[dict]:
        def _query() -> Optional[dict]:
            cur = self._conn.execute(
                "SELECT * FROM posts WHERE post_id = ? LIMIT 1",
                (post_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            post = dict(row)
            image_urls = json.loads(post.get("images") or "[]")
            images_cur = self._conn.execute(
                """
                SELECT position, url, content_type, data
                FROM post_images
                WHERE post_id = ?
                ORDER BY position ASC
                """,
                (post_id,),
            )
            image_files = []
            for image_row in images_cur.fetchall():
                data_blob = image_row[3]
                encoded = (
                    base64.b64encode(data_blob).decode("ascii") if data_blob is not None else ""
                )
                image_files.append(
                    {
                        "url": image_row[1] or (image_urls[image_row[0]] if image_row[0] < len(image_urls) else None),
                        "content_type": image_row[2],
                        "data": encoded,
                    }
                )
            post["image_urls"] = image_urls
            post["images"] = image_files
            comments_cur = self._conn.execute(
                """
                SELECT comment_id, parent_id, nickname, user_id, ip, created_at, content, dccons, is_deleted
                FROM comments
                WHERE post_id = ?
                ORDER BY created_at ASC
                """,
                (post_id,),
            )
            ordered_comments = []
            comment_map: dict[str, dict] = {}
            for row in comments_cur.fetchall():
                comment = dict(row)
                comment_id = comment.get("comment_id")
                if isinstance(comment_id, str):
                    comment_id = comment_id.strip()
                    comment["comment_id"] = comment_id
                dccons = json.loads(comment.get("dccons") or "[]")
                comment["dccons"] = dccons
                comment.setdefault("images", dccons)
                comment.setdefault("replies", [])
                if comment_id is not None and "id" not in comment:
                    comment["id"] = comment_id
                    comment_map[str(comment_id)] = comment
                ordered_comments.append(comment)

            threaded: list[dict] = []
            for comment in ordered_comments:
                parent_id = comment.get("parent_id")
                if isinstance(parent_id, str):
                    parent_id = parent_id.strip()
                    comment["parent_id"] = parent_id
                if parent_id in (None, "", "0"):
                    threaded.append(comment)
                    continue

                parent = comment_map.get(str(parent_id))
                if parent is None or parent is comment:
                    threaded.append(comment)
                    continue

                replies = parent.setdefault("replies", [])
                replies.append(comment)

            post["comments"] = threaded
            return post

        return await asyncio.to_thread(_query)


__all__ = ["GalleryDatabase", "StoredComment"]
