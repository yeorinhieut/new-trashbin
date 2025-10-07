"""High level orchestration for the DCInside archiver."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tomllib
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tomlkit import aot, document, dumps, table

from .dcinside import (
    Author,
    DCInsideClient,
    Comment,
    PostDeletedError,
    PostDetail,
    PostSummary,
    RateLimitedError,
)
from .llm import DEFAULT_SUMMARY_PROMPT
from .db import GalleryDatabase
from .queue import QueueDatabase, QueueRow

LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path("config.toml")


class GallerySettings(BaseModel):
    id: str
    name: Optional[str] = None
    delay: float = Field(default=30.0, ge=1.0)
    fetch_delay: float = Field(default=1.0, ge=0.0)
    pages: int = Field(default=1, ge=1, le=10)
    followup_delays: List[float] = Field(default_factory=lambda: [60, 300, 1800])
    auto_start: bool = True
    max_comment_pages: int = Field(default=5, ge=1)
    comment_page_size: int = Field(default=100, ge=10)
    comment_delay: float = Field(default=0.5, ge=0.0)
    comment_followup_delay: float = Field(default=1800.0, ge=0.0)
    comment_followup_threshold: float = Field(default=300.0, ge=0.0)
    comment_followup_max_retries: int = Field(default=3, ge=0)

    @field_validator("followup_delays", mode="before")
    @classmethod
    def _ensure_sorted(cls, value: Iterable[float]) -> List[float]:
        delays = [float(v) for v in value]
        return sorted(delays)


class ArchiveOptions(BaseModel):
    base_url: str = "https://m.dcinside.com"
    user_agent: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class CommentOptions(BaseModel):
    max_pages: int = Field(default=5, ge=1)
    page_size: int = Field(default=100, ge=10)
    delay: float = Field(default=0.5, ge=0.0)


class LLMOptions(BaseModel):
    api_key: Optional[str] = None
    model: str = "gpt5"
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT


class ArchiveConfigModel(BaseModel):
    archive: ArchiveOptions = ArchiveOptions()
    comments: CommentOptions = CommentOptions()
    galleries: List[GallerySettings] = Field(default_factory=list)
    llm: LLMOptions = LLMOptions()


@dataclass
class ArchiveConfig:
    path: Path
    model: ArchiveConfigModel

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "ArchiveConfig":
        path = Path(path)
        if path.exists():
            raw = tomllib.loads(path.read_text("utf-8"))
        else:
            raw = {}
        model = ArchiveConfigModel(**raw)
        if not model.galleries:
            # Seed with a default gallery to guide the user.
            model.galleries.append(GallerySettings(id="chatgpt", name="ChatGPT", auto_start=False))
        return cls(path=path, model=model)

    def save(self) -> None:
        doc = document()
        archive_table = table()
        archive_table["base_url"] = self.model.archive.base_url
        if self.model.archive.user_agent:
            archive_table["user_agent"] = self.model.archive.user_agent
        doc["archive"] = archive_table

        comments_table = table()
        comments_table["max_pages"] = self.model.comments.max_pages
        comments_table["page_size"] = self.model.comments.page_size
        comments_table["delay"] = self.model.comments.delay
        doc["comments"] = comments_table

        llm_table = table()
        llm_options = self.model.llm
        if llm_options.api_key is not None:
            llm_table["api_key"] = llm_options.api_key or ""
        llm_table["model"] = llm_options.model
        llm_table["summary_prompt"] = llm_options.summary_prompt
        doc["llm"] = llm_table

        gallery_array = aot()
        for gallery in self.model.galleries:
            entry = table()
            entry["id"] = gallery.id
            if gallery.name:
                entry["name"] = gallery.name
            entry["delay"] = gallery.delay
            entry["fetch_delay"] = gallery.fetch_delay
            entry["pages"] = gallery.pages
            entry["followup_delays"] = list(gallery.followup_delays)
            entry["auto_start"] = gallery.auto_start
            entry["max_comment_pages"] = gallery.max_comment_pages
            entry["comment_page_size"] = gallery.comment_page_size
            entry["comment_delay"] = gallery.comment_delay
            entry["comment_followup_delay"] = gallery.comment_followup_delay
            entry["comment_followup_threshold"] = gallery.comment_followup_threshold
            entry["comment_followup_max_retries"] = gallery.comment_followup_max_retries
            gallery_array.append(entry)
        doc["galleries"] = gallery_array

        self.path.write_text(dumps(doc), encoding="utf-8")

    def get_gallery(self, gallery_id: str) -> Optional[GallerySettings]:
        for gallery in self.model.galleries:
            if gallery.id == gallery_id:
                return gallery
        return None

    def update_gallery(self, gallery_id: str, **changes: Any) -> GallerySettings:
        for index, gallery in enumerate(self.model.galleries):
            if gallery.id == gallery_id:
                updated = gallery.model_copy(update=changes)
                self.model.galleries[index] = updated
                return updated
        raise KeyError(f"Gallery {gallery_id} not found")

    def add_gallery(self, settings: GallerySettings) -> GallerySettings:
        if self.get_gallery(settings.id):
            raise ValueError(f"Gallery {settings.id} already exists")
        self.model.galleries.append(settings)
        self.save()
        return settings


@dataclass
class ArchiveEvent:
    type: str
    gallery_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GalleryStatus:
    gallery_id: str
    name: Optional[str]
    running: bool
    total_posts: int
    last_activity: Optional[datetime]
    last_error: Optional[str]
    queue_size: int
    queue: List[QueueEntry]
    delay: float
    fetch_delay: float
    pages: int
    followup_delays: List[float]
    auto_start: bool
    max_comment_pages: int
    comment_page_size: int
    comment_delay: float
    comment_followup_delay: float
    comment_followup_threshold: float
    comment_followup_max_retries: int
    db_size_bytes: int


@dataclass
class PostListItem:
    post_id: str
    title: Optional[str]
    nickname: Optional[str]
    user_id: Optional[str]
    created_at: Optional[str]
    view_count: Optional[int]
    recommend_count: Optional[int]
    dislike_count: Optional[int]
    is_deleted: bool


@dataclass
class PostPage:
    items: List[PostListItem]
    page: int
    page_size: int
    total: int

    @property
    def total_pages(self) -> int:
        if self.total == 0:
            return 1
        return max((self.total + self.page_size - 1) // self.page_size, 1)


DownloadedImage = Tuple[str, bytes, Optional[str]]


@dataclass
class QueueEntry:
    gallery_id: str
    post_id: str
    title: Optional[str]
    state: str = "queued"


@dataclass
class QueueSnapshotItem:
    queue_id: int
    gallery_id: str
    post_id: str
    action: str
    state: str
    manual: bool
    available_at: float


@dataclass
class QueueOverview:
    collecting_total: int
    collecting: List[QueueSnapshotItem]
    scheduled_total: int
    scheduled: List[QueueSnapshotItem]


@dataclass
class QueueWorkItem:
    queue_id: int
    post_id: str
    action: str
    delay: float = 0.0
    title: Optional[str] = None
    summary: Optional[PostSummary] = None
    manual: bool = False
    available_at: float = 0.0


class GalleryArchiver:
    def __init__(
        self,
        settings: GallerySettings,
        manager: "ArchiveManager",
        *,
        database: GalleryDatabase,
        queue: QueueDatabase,
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.database = database
        self.queue = queue
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._followups: dict[str, set[int]] = {}
        self._known_posts: set[str] = set()
        self.total_posts = 0
        self.last_activity: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()
        self._queue_event = asyncio.Event()
        self._queue_task: Optional[asyncio.Task] = None
        self._pending_new_posts: set[str] = set()
        self._current_item: Optional[QueueWorkItem] = None
        self._comment_followup_attempts: dict[str, int] = {}
        self._post_titles: dict[str, str] = {}

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._known_posts = await self.database.known_post_ids()
        self.total_posts = len(self._known_posts)
        await self.queue.reset_processing(self.settings.id)
        pending_new = await self.queue.pending_post_ids(self.settings.id, "new")
        titles = await self.queue.titles_for_gallery(self.settings.id)
        followups = await self.queue.load_followups(self.settings.id)
        comment_counts = await self.queue.comment_followup_counts(self.settings.id)
        async with self._queue_lock:
            self._pending_new_posts = set(pending_new)
            self._current_item = None
            self._queue_event.clear()
            for post_id, title in titles.items():
                self._post_titles[post_id] = title
        self._followups = {post_id: set(ids) for post_id, ids in followups.items()}
        self._comment_followup_attempts = dict(comment_counts)
        if not self._queue_task or self._queue_task.done():
            self._queue_task = asyncio.create_task(
                self._process_queue(), name=f"queue:{self.settings.id}"
            )
        self._task = asyncio.create_task(self._run(), name=f"gallery:{self.settings.id}")
        self._queue_event.set()

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._followups.clear()
        if self._queue_task:
            self._queue_event.set()
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass
            self._queue_task = None
        async with self._queue_lock:
            self._pending_new_posts.clear()
            self._current_item = None
            self._queue_event.clear()
        self._comment_followup_attempts.clear()

    async def poll_once(self) -> None:
        try:
            await self._poll_once()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Polling error for %s", self.settings.id)
            self.last_error = str(exc)
            await self.manager.emit(
                ArchiveEvent(
                    type="error",
                    gallery_id=self.settings.id,
                    payload={"error": str(exc)},
                )
            )

    async def _run(self) -> None:
        while self.running:
            await self.poll_once()
            await asyncio.sleep(self.settings.delay)

    async def _poll_once(self) -> None:
        async with self._lock:
            try:
                summaries = await self.manager.client.fetch_board_page(
                    self.settings.id,
                    page=1,
                )
            except RateLimitedError as exc:
                self.last_error = str(exc)
                await asyncio.sleep(exc.retry_after)
                return
            for summary in summaries:
                await self._enqueue_post(summary)
        self.last_activity = datetime.now(timezone.utc)

    async def _enqueue_post(
        self, summary: PostSummary, *, front: bool = False
    ) -> None:
        if summary.post_id in self._known_posts:
            return
        async with self._queue_lock:
            if summary.post_id in self._pending_new_posts:
                return
            self._pending_new_posts.add(summary.post_id)
            self._post_titles[summary.post_id] = summary.title
        payload = self._summary_to_payload(summary)
        await self.queue.enqueue(
            self.settings.id,
            summary.post_id,
            "new",
            title=summary.title,
            manual=False,
            delay=0.0,
            payload=payload,
            front=front,
        )
        self._queue_event.set()

    async def _process_queue(self) -> None:
        try:
            while self.running:
                row, wake_at = await self.queue.acquire_next(self.settings.id)
                if not self.running:
                    break
                if row is None:
                    timeout: Optional[float] = None
                    if wake_at is not None:
                        timeout = max(0.0, wake_at - time.time())
                    try:
                        if timeout is None:
                            await self._queue_event.wait()
                        else:
                            await asyncio.wait_for(self._queue_event.wait(), timeout)
                    except asyncio.TimeoutError:
                        pass
                    finally:
                        self._queue_event.clear()
                    continue
                item = self._row_to_work_item(row)
                async with self._queue_lock:
                    self._current_item = item
                reset_pending = item.action == "new"
                complete_item = True
                try:
                    await self._handle_queue_item(item)
                except RateLimitedError as exc:
                    complete_item = False
                    self.last_error = str(exc)
                    await asyncio.sleep(exc.retry_after)
                    await self.queue.reschedule(item.queue_id, delay=0.0, front=True)
                    self._queue_event.set()
                    reset_pending = False
                except Exception as exc:  # pragma: no cover - defensive
                    if item.action == "new":
                        async with self._queue_lock:
                            self._pending_new_posts.discard(item.post_id)
                    LOGGER.exception(
                        "Failed to process queued item %s/%s (%s)",
                        self.settings.id,
                        item.post_id,
                        item.action,
                    )
                    self.last_error = str(exc)
                    if self.settings.fetch_delay > 0 and self.running:
                        await asyncio.sleep(self.settings.fetch_delay)
                else:
                    if item.action == "new":
                        async with self._queue_lock:
                            self._pending_new_posts.discard(item.post_id)
                    self.last_error = None
                    if self.settings.fetch_delay > 0 and self.running:
                        await asyncio.sleep(self.settings.fetch_delay)
                finally:
                    async with self._queue_lock:
                        if self._current_item is item:
                            self._current_item = None
                    if complete_item:
                        await self.queue.complete(item.queue_id)
                        if (
                            item.action in {"followup", "comment_followup"}
                            and not item.manual
                        ):
                            self._remove_followup_entry(item.post_id, item.queue_id)
                    if item.action == "new" and reset_pending:
                        async with self._queue_lock:
                            self._pending_new_posts.discard(item.post_id)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass

    async def _handle_queue_item(self, item: QueueWorkItem) -> None:
        if item.action == "new":
            if not item.summary:
                return
            await self._handle_new_post(item.summary)
            return
        source = item.action
        if source == "manual":
            await self._followup(item.post_id, item.delay, manual=True, source="manual")
        elif source == "comment_followup":
            await self._followup(
                item.post_id,
                item.delay,
                manual=False,
                source="comment_followup",
            )
        else:
            await self._followup(item.post_id, item.delay, manual=False, source="followup")

    def _queue_entry_for_item(self, item: QueueWorkItem, *, state: str) -> QueueEntry:
        title = item.title or self._post_titles.get(item.post_id)
        if item.action == "comment_followup":
            title = self._prefixed_title(title, "[댓글]")
        elif item.action == "manual":
            title = self._prefixed_title(title, "[재수집]")
        elif item.action == "followup":
            title = self._prefixed_title(title, "[후속]")
        return QueueEntry(
            gallery_id=self.settings.id,
            post_id=item.post_id,
            title=title,
            state=state,
        )

    def _queue_entry_for_row(self, row: QueueRow, *, state: str) -> QueueEntry:
        title = row.title or self._post_titles.get(row.post_id)
        action = row.action
        if action == "comment_followup":
            title = self._prefixed_title(title, "[댓글]")
        elif action == "manual":
            title = self._prefixed_title(title, "[재수집]")
        elif action == "followup":
            title = self._prefixed_title(title, "[후속]")
        return QueueEntry(
            gallery_id=row.gallery_id,
            post_id=row.post_id,
            title=title,
            state=state,
        )

    @staticmethod
    def _prefixed_title(title: Optional[str], prefix: str) -> str:
        if title:
            return f"{prefix} {title}"
        return prefix

    def _row_to_work_item(self, row: QueueRow) -> QueueWorkItem:
        summary: Optional[PostSummary] = None
        if row.action == "new" and row.payload:
            try:
                summary = self._summary_from_payload(row.payload)
            except Exception:  # pragma: no cover - defensive
                summary = None
        title = row.title or self._post_titles.get(row.post_id)
        if not title and summary:
            title = summary.title
        if title:
            self._post_titles[row.post_id] = title
        return QueueWorkItem(
            queue_id=row.id,
            post_id=row.post_id,
            action=row.action,
            delay=row.delay,
            title=title,
            summary=summary,
            manual=row.manual,
            available_at=row.available_at,
        )

    @staticmethod
    def _summary_to_payload(summary: PostSummary) -> Dict[str, Any]:
        return {
            "gallery_id": summary.gallery_id,
            "post_id": summary.post_id,
            "type": summary.type,
            "subject": summary.subject,
            "title": summary.title,
            "link": summary.link,
            "author": {
                "nickname": summary.author.nickname,
                "user_id": summary.author.user_id,
                "ip": summary.author.ip,
            },
            "created_at": summary.created_at,
            "view_count": summary.view_count,
            "recommend_count": summary.recommend_count,
            "comment_count": summary.comment_count,
        }

    @staticmethod
    def _summary_from_payload(payload: Dict[str, Any]) -> PostSummary:
        author_payload = payload.get("author", {})
        author = Author(
            nickname=str(author_payload.get("nickname", "")),
            user_id=author_payload.get("user_id"),
            ip=author_payload.get("ip"),
        )
        return PostSummary(
            gallery_id=str(payload.get("gallery_id", "")),
            post_id=str(payload.get("post_id", "")),
            type=str(payload.get("type", "")),
            subject=str(payload.get("subject", "")),
            title=str(payload.get("title", "")),
            link=str(payload.get("link", "")),
            author=author,
            created_at=str(payload.get("created_at", "")),
            view_count=int(payload.get("view_count", 0) or 0),
            recommend_count=int(payload.get("recommend_count", 0) or 0),
            comment_count=int(payload.get("comment_count", 0) or 0),
        )

    def _remove_followup_entry(self, post_id: str, queue_id: int) -> None:
        entries = self._followups.get(post_id)
        if not entries:
            return
        entries.discard(queue_id)
        if entries:
            self._followups[post_id] = entries
        else:
            self._followups.pop(post_id, None)

    async def queue_state(self, limit: int = 10) -> tuple[int, List[QueueEntry]]:
        async with self._queue_lock:
            current = self._current_item
        due_count = await self.queue.count_due(self.settings.id)
        queue_size = due_count + (1 if current else 0)
        preview: list[QueueEntry] = []
        if current:
            preview.append(self._queue_entry_for_item(current, state="processing"))
        remaining = max(0, limit - len(preview))
        if remaining > 0:
            rows = await self.queue.list_due(self.settings.id, remaining)
            for row in rows:
                if row.title:
                    self._post_titles[row.post_id] = row.title
                preview.append(self._queue_entry_for_row(row, state="queued"))
        return queue_size, preview

    async def _handle_new_post(self, summary: PostSummary) -> None:
        try:
            detail = await self.manager.client.fetch_post(
                summary.gallery_id,
                summary.post_id,
                extract_images=True,
                include_image_source=False,
                fetch_comments=True,
                max_comment_pages=self.settings.max_comment_pages,
                comment_page_size=self.settings.comment_page_size,
                comment_delay=self.settings.comment_delay,
            )
        except PostDeletedError as exc:
            LOGGER.info("Skipping removed post %s/%s: %s", summary.gallery_id, summary.post_id, exc)
            return
        except RateLimitedError:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to fetch post %s/%s", summary.gallery_id, summary.post_id)
            self.last_error = str(exc)
            return

        if detail is None:
            return

        author = detail.author
        summary_author = summary.author
        if summary_author.user_id and not author.user_id:
            author.user_id = summary_author.user_id
        if summary_author.nickname and not author.nickname:
            author.nickname = summary_author.nickname
        if summary_author.ip and not author.ip:
            author.ip = summary_author.ip
        if author.nickname:
            author.nickname = author.nickname.strip()
        if author.ip and author.nickname and not author.nickname.endswith(f"({author.ip})"):
            author.nickname = f"{author.nickname}({author.ip})"

        images = await self._download_images(detail.images)

        await self.database.upsert_post(detail, images=images)
        if detail.title:
            self._post_titles[summary.post_id] = detail.title
        self._known_posts.add(summary.post_id)
        self.total_posts += 1
        self.last_error = None
        await self.manager.emit(
            ArchiveEvent(
                type="post.new",
                gallery_id=summary.gallery_id,
                payload={
                    "post_id": summary.post_id,
                    "title": detail.title,
                    "author": detail.author.nickname,
                },
            )
        )
        await self._schedule_followups(summary.post_id)

    async def _schedule_followups(self, post_id: str) -> None:
        entries: set[int] = set()
        for delay in self.settings.followup_delays:
            row = await self._queue_followup(
                post_id,
                delay=delay,
                source="followup",
                manual=False,
            )
            entries.add(row.id)
        if entries:
            self._followups[post_id] = entries
            self._comment_followup_attempts[post_id] = 0

    async def recrawl_post(self, post_id: str) -> None:
        await self._queue_followup(
            post_id,
            delay=0.0,
            source="manual",
            manual=True,
            front=True,
        )

    async def _queue_followup(
        self,
        post_id: str,
        *,
        delay: float,
        source: str,
        manual: bool,
        front: bool = False,
    ) -> QueueRow:
        title = self._post_titles.get(post_id)
        if not title:
            existing = await self.database.fetch_post(post_id)
            if existing and existing.get("title"):
                title = str(existing.get("title"))
                self._post_titles[post_id] = title
        action = source if source != "manual" else "manual"
        row = await self.queue.enqueue(
            self.settings.id,
            post_id,
            action,
            title=title,
            manual=manual,
            delay=delay,
            front=front,
        )
        if not manual and action in {"followup", "comment_followup"}:
            entries = self._followups.setdefault(post_id, set())
            entries.add(row.id)
            self._followups[post_id] = entries
        if row.title:
            self._post_titles[post_id] = row.title
        self._queue_event.set()
        return row

    async def _followup(
        self,
        post_id: str,
        delay: float,
        *,
        manual: bool = False,
        source: str = "followup",
    ) -> None:
        try:
            detail, images = await self._fetch_post_detail_with_retry(post_id)
            if not detail:
                return
            if detail.is_deleted:
                await self.database.update_counts(post_id, is_deleted=True)
                await self.queue.delete_for_post(self.settings.id, post_id)
                self._followups.pop(post_id, None)
                self._comment_followup_attempts.pop(post_id, None)
                await self.manager.emit(
                    ArchiveEvent(
                        type="post.deleted",
                        gallery_id=self.settings.id,
                        payload={"post_id": post_id, "delay": delay},
                    )
                )
                return

            existing = await self.database.fetch_post(post_id)
            author = detail.author
            if existing:
                if not author.user_id and existing.get("user_id"):
                    author.user_id = existing.get("user_id")
                if (not author.nickname or author.nickname == "익명") and existing.get("nickname"):
                    author.nickname = existing.get("nickname")
                if not author.ip and existing.get("ip"):
                    author.ip = existing.get("ip")
            if author.nickname:
                author.nickname = author.nickname.strip()
            if author.ip and author.nickname and not author.nickname.endswith(f"({author.ip})"):
                author.nickname = f"{author.nickname}({author.ip})"
            if detail.title:
                self._post_titles[post_id] = detail.title
            await self.database.upsert_post(detail, images=images)
            await self.manager.emit(
                ArchiveEvent(
                    type="post.update",
                    gallery_id=self.settings.id,
                    payload={
                        "post_id": post_id,
                        "delay": delay,
                    },
                )
            )
            self.last_error = None

            if (
                source != "comment_followup"
                and self.settings.comment_followup_delay >= 0
            ):
                await self._maybe_schedule_comment_followup(post_id, detail)
        except PostDeletedError as exc:
            LOGGER.info(
                "Marking post %s/%s as deleted after %ss: %s",
                self.settings.id,
                post_id,
                delay,
                exc,
            )
            await self.database.update_counts(post_id, is_deleted=True)
            await self.queue.delete_for_post(self.settings.id, post_id)
            self._followups.pop(post_id, None)
            await self.manager.emit(
                ArchiveEvent(
                    type="post.deleted",
                    gallery_id=self.settings.id,
                    payload={"post_id": post_id, "delay": delay},
                )
            )
            self._comment_followup_attempts.pop(post_id, None)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Follow-up failed for %s/%s", self.settings.id, post_id)
            self.last_error = str(exc)
        finally:
            if source == "comment_followup":
                attempts = self._comment_followup_attempts.get(post_id, 0)
                if attempts > 1:
                    self._comment_followup_attempts[post_id] = attempts - 1
                else:
                    self._comment_followup_attempts.pop(post_id, None)
            elif not manual and not self._followups.get(post_id):
                self._comment_followup_attempts.pop(post_id, None)

    async def _fetch_post_detail_with_retry(
        self, post_id: str
    ) -> tuple[Optional[PostDetail], list[DownloadedImage]]:
        while True:
            try:
                detail = await self.manager.client.fetch_post(
                    self.settings.id,
                    post_id,
                    extract_images=True,
                    include_image_source=False,
                    fetch_comments=True,
                    max_comment_pages=self.settings.max_comment_pages,
                    comment_page_size=self.settings.comment_page_size,
                    comment_delay=self.settings.comment_delay,
                )
                images: list[DownloadedImage] = []
                if detail:
                    images = await self._download_images(detail.images)
                return detail, images
            except RateLimitedError as exc:
                self.last_error = str(exc)
                await asyncio.sleep(exc.retry_after)

    async def _maybe_schedule_comment_followup(
        self, post_id: str, detail: PostDetail
    ) -> None:
        if not self.running:
            return
        max_retries = self.settings.comment_followup_max_retries
        threshold = self.settings.comment_followup_threshold
        if max_retries <= 0 or threshold < 0:
            self._comment_followup_attempts.pop(post_id, None)
            return
        if not detail.comments:
            self._comment_followup_attempts.pop(post_id, None)
            return
        latest_comment = self._extract_latest_comment_time(detail.comments)
        if not latest_comment:
            self._comment_followup_attempts.pop(post_id, None)
            return
        now = datetime.now(timezone.utc)
        diff_seconds = (now - latest_comment.astimezone(timezone.utc)).total_seconds()
        if diff_seconds < 0:
            diff_seconds = 0.0
        if diff_seconds <= threshold:
            attempts = self._comment_followup_attempts.get(post_id, 0)
            if attempts >= max_retries:
                return
            delay = max(self.settings.comment_followup_delay, 0.0)
            if delay <= 0:
                return
            await self._queue_followup(
                post_id,
                delay=delay,
                source="comment_followup",
                manual=False,
            )
            self._comment_followup_attempts[post_id] = attempts + 1
        else:
            self._comment_followup_attempts.pop(post_id, None)

    def _extract_latest_comment_time(
        self, comments: Sequence[Comment]
    ) -> Optional[datetime]:
        latest: Optional[datetime] = None
        for comment in comments:
            candidate = self._parse_comment_timestamp(comment.created_at)
            if not candidate:
                continue
            if latest is None or candidate > latest:
                latest = candidate
        return latest

    def _parse_comment_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        normalized = text
        if "T" not in normalized and " " in normalized:
            normalized = normalized.replace(" ", "T", 1)
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def _download_images(
        self, image_urls: Sequence[str]
    ) -> list[DownloadedImage]:
        images: list[DownloadedImage] = []
        for url in image_urls:
            if not url:
                continue
            try:
                content, content_type = await self.manager.client.download_image(url)
            except RateLimitedError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Failed to download image for %s (%s): %s",
                    self.settings.id,
                    url,
                    exc,
                )
                continue
            images.append((url, content, content_type))
        return images


class ArchiveManager:
    def __init__(self, config: ArchiveConfig) -> None:
        self.config = config
        self.client = DCInsideClient(
            base_url=config.model.archive.base_url,
            user_agent=config.model.archive.user_agent,
        )
        self._databases: dict[str, GalleryDatabase] = {}
        self._galleries: dict[str, GalleryArchiver] = {}
        self._events: asyncio.Queue[ArchiveEvent] = asyncio.Queue()
        self.queue = QueueDatabase()

        for gallery in self.config.model.galleries:
            self._galleries[gallery.id] = self._create_archiver(gallery)

    def _create_archiver(self, settings: GallerySettings) -> GalleryArchiver:
        database = self._databases.get(settings.id)
        if not database:
            database = GalleryDatabase(settings.id)
            self._databases[settings.id] = database
        return GalleryArchiver(settings, self, database=database, queue=self.queue)

    async def close(self) -> None:
        for archiver in self._galleries.values():
            await archiver.stop()
        await self.client.aclose()
        for db in self._databases.values():
            await db.close()
        await self.queue.close()

    async def start_auto(self) -> None:
        for archiver in self._galleries.values():
            if archiver.settings.auto_start:
                await archiver.start()

    async def start_gallery(self, gallery_id: str) -> None:
        archiver = self._galleries.get(gallery_id)
        if archiver:
            await archiver.start()

    async def stop_gallery(self, gallery_id: str) -> None:
        archiver = self._galleries.get(gallery_id)
        if archiver:
            await archiver.stop()

    async def recrawl_post(self, gallery_id: str, post_id: str) -> None:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        await archiver.recrawl_post(post_id)

    async def reload(self) -> None:
        await self.client.aclose()
        self.config = ArchiveConfig.load(self.config.path)
        self.client = DCInsideClient(
            base_url=self.config.model.archive.base_url,
            user_agent=self.config.model.archive.user_agent,
        )
        for gallery_id, archiver in list(self._galleries.items()):
            await archiver.stop()
        self._galleries.clear()
        for gallery in self.config.model.galleries:
            self._galleries[gallery.id] = self._create_archiver(gallery)
        await self.start_auto()

    async def fetch_posts_for_summary(
        self,
        gallery_id: str,
        *,
        limit: int = 50,
        search: Optional[str] = None,
        author: Optional[str] = None,
        user_id: Optional[str] = None,
        deleted_only: bool = False,
    ) -> List[dict]:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows = await archiver.database.fetch_posts_for_summary(
            limit=limit,
            search=search,
            author=author,
            user_id=user_id,
            deleted_only=deleted_only,
        )
        results: list[dict] = []
        for row in rows:
            results.append(dict(row))
        return results

    async def emit(self, event: ArchiveEvent) -> None:
        await self._events.put(event)

    async def get_event(self) -> ArchiveEvent:
        return await self._events.get()

    def get_archiver(self, gallery_id: str) -> Optional[GalleryArchiver]:
        return self._galleries.get(gallery_id)

    def iter_archivers(self) -> Iterable[GalleryArchiver]:
        return self._galleries.values()

    async def get_status(self, gallery_id: str) -> GalleryStatus:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        queue_size, queue_preview = await archiver.queue_state()
        settings = archiver.settings
        return GalleryStatus(
            gallery_id=archiver.settings.id,
            name=archiver.settings.name,
            running=archiver.running,
            total_posts=archiver.total_posts,
            last_activity=archiver.last_activity,
            last_error=archiver.last_error,
            queue_size=queue_size,
            queue=queue_preview,
            delay=settings.delay,
            fetch_delay=settings.fetch_delay,
            pages=settings.pages,
            followup_delays=list(settings.followup_delays),
            auto_start=settings.auto_start,
            max_comment_pages=settings.max_comment_pages,
            comment_page_size=settings.comment_page_size,
            comment_delay=settings.comment_delay,
            comment_followup_delay=settings.comment_followup_delay,
            comment_followup_threshold=settings.comment_followup_threshold,
            comment_followup_max_retries=settings.comment_followup_max_retries,
            db_size_bytes=archiver.database.file_size(),
        )

    async def status_snapshot(self) -> List[GalleryStatus]:
        statuses: list[GalleryStatus] = []
        for archiver in self._galleries.values():
            queue_size, queue_preview = await archiver.queue_state()
            settings = archiver.settings
            statuses.append(
                GalleryStatus(
                    gallery_id=archiver.settings.id,
                    name=archiver.settings.name,
                    running=archiver.running,
                    total_posts=archiver.total_posts,
                    last_activity=archiver.last_activity,
                    last_error=archiver.last_error,
                    queue_size=queue_size,
                    queue=queue_preview,
                    delay=settings.delay,
                    fetch_delay=settings.fetch_delay,
                    pages=settings.pages,
                    followup_delays=list(settings.followup_delays),
                    auto_start=settings.auto_start,
                    max_comment_pages=settings.max_comment_pages,
                    comment_page_size=settings.comment_page_size,
                    comment_delay=settings.comment_delay,
                    comment_followup_delay=settings.comment_followup_delay,
                    comment_followup_threshold=settings.comment_followup_threshold,
                    comment_followup_max_retries=settings.comment_followup_max_retries,
                    db_size_bytes=archiver.database.file_size(),
                )
            )
        return statuses

    async def queue_overview(self, *, limit: int = 10) -> QueueOverview:
        collecting_total = await self.queue.count_global_due()
        scheduled_total = await self.queue.count_global_scheduled()
        collecting_rows = await self.queue.list_global_due(limit)
        scheduled_rows = await self.queue.list_global_scheduled(limit)
        return QueueOverview(
            collecting_total=collecting_total,
            collecting=[
                self._row_to_snapshot_item(row) for row in collecting_rows
            ],
            scheduled_total=scheduled_total,
            scheduled=[
                self._row_to_snapshot_item(row) for row in scheduled_rows
            ],
        )

    @staticmethod
    def _row_to_snapshot_item(row: QueueRow) -> QueueSnapshotItem:
        return QueueSnapshotItem(
            queue_id=row.id,
            gallery_id=row.gallery_id,
            post_id=row.post_id,
            action=row.action,
            state=row.state,
            manual=row.manual,
            available_at=row.available_at,
        )

    async def add_gallery(self, settings: GallerySettings) -> GalleryStatus:
        if settings.id in self._galleries:
            raise ValueError(f"Gallery {settings.id} already exists")
        self.config.add_gallery(settings)
        archiver = self._create_archiver(settings)
        self._galleries[settings.id] = archiver
        if settings.auto_start:
            await archiver.start()
        return await self.get_status(settings.id)

    async def update_gallery(self, gallery_id: str, **changes: Any) -> GalleryStatus:
        updated = self.config.update_gallery(gallery_id, **changes)
        self.config.save()
        archiver = self._galleries.get(gallery_id)
        if archiver:
            archiver.settings = updated
        else:
            archiver = self._create_archiver(updated)
            self._galleries[gallery_id] = archiver
        return await self.get_status(gallery_id)

    async def search_posts(self, gallery_id: str, query: str, *, limit: int = 25) -> List[dict]:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows = await archiver.database.search_posts(query, limit=limit)
        results = []
        for row in rows:
            results.append({
                "post_id": row["post_id"],
                "title": row["title"],
                "nickname": row["nickname"],
                "created_at": row["created_at"],
                "view_count": row["view_count"],
                "recommend_count": row["recommend_count"],
            })
        return results

    async def list_posts(
        self,
        gallery_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> PostPage:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows, total = await archiver.database.list_posts(
            page=page, page_size=page_size, deleted_only=deleted_only
        )
        return PostPage(
            items=[
                PostListItem(
                    post_id=row["post_id"],
                    title=row["title"],
                    nickname=row["nickname"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    view_count=row["view_count"],
                    recommend_count=row["recommend_count"],
                    dislike_count=row["dislike_count"],
                    is_deleted=bool(row["is_deleted"]),
                )
                for row in rows
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def search_posts_paginated(
        self,
        gallery_id: str,
        query: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> PostPage:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows, total = await archiver.database.search_posts_paginated(
            query,
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )
        return PostPage(
            items=[
                PostListItem(
                    post_id=row["post_id"],
                    title=row["title"],
                    nickname=row["nickname"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    view_count=row["view_count"],
                    recommend_count=row["recommend_count"],
                    dislike_count=row["dislike_count"],
                    is_deleted=bool(row["is_deleted"]),
                )
                for row in rows
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def search_posts_by_author(
        self,
        gallery_id: str,
        nickname: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> PostPage:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows, total = await archiver.database.search_posts_by_author(
            nickname,
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )
        return PostPage(
            items=[
                PostListItem(
                    post_id=row["post_id"],
                    title=row["title"],
                    nickname=row["nickname"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    view_count=row["view_count"],
                    recommend_count=row["recommend_count"],
                    dislike_count=row["dislike_count"],
                    is_deleted=bool(row["is_deleted"]),
                )
                for row in rows
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def search_posts_by_user_id(
        self,
        gallery_id: str,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        deleted_only: bool = False,
    ) -> PostPage:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        rows, total = await archiver.database.search_posts_by_user_id(
            user_id,
            page=page,
            page_size=page_size,
            deleted_only=deleted_only,
        )
        return PostPage(
            items=[
                PostListItem(
                    post_id=row["post_id"],
                    title=row["title"],
                    nickname=row["nickname"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    view_count=row["view_count"],
                    recommend_count=row["recommend_count"],
                    dislike_count=row["dislike_count"],
                    is_deleted=bool(row["is_deleted"]),
                )
                for row in rows
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def fetch_post(self, gallery_id: str, post_id: str) -> Optional[dict]:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        return await archiver.database.fetch_post(post_id)

    async def export_post(self, gallery_id: str, post_id: str, *, output_dir: Path | str = "exports") -> Path:
        archiver = self._galleries.get(gallery_id)
        if not archiver:
            raise KeyError(f"Gallery {gallery_id} not found")
        post = await archiver.database.fetch_post(post_id)
        if not post:
            raise KeyError(f"Post {post_id} not found")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{gallery_id}_{post_id}.json"
        output_path.write_text(json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def update_gallery_setting(self, gallery_id: str, field_name: str, value: Any) -> GallerySettings:
        updated = self.config.update_gallery(gallery_id, **{field_name: value})
        self.config.save()
        archiver = self._galleries.get(gallery_id)
        if archiver:
            archiver.settings = updated
        else:
            self._galleries[gallery_id] = self._create_archiver(updated)
        return updated


__all__ = [
    "ArchiveConfig",
    "ArchiveManager",
    "GallerySettings",
    "ArchiveEvent",
    "GalleryStatus",
    "PostListItem",
    "PostPage",
    "QueueEntry",
    "QueueOverview",
    "QueueSnapshotItem",
    "LLMOptions",
]

