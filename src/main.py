"""FastAPI application exposing the DCInside archiver controls and dashboard."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import time

import tomllib
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .archive import (
    ArchiveConfig,
    ArchiveConfigModel,
    ArchiveManager,
    GallerySettings,
    GalleryStatus,
    PostListItem,
    PostPage,
)
from .llm import (
    LLMConfigurationError,
    LLMError,
    LLMPost,
    LLMService,
)

DASHBOARD_PATH = Path(__file__).resolve().parent / "static" / "index.html"
VIEW_PATH = Path(__file__).resolve().parent / "static" / "view.html"
LIST_PATH = Path(__file__).resolve().parent / "static" / "list.html"


class QueueItemResponse(BaseModel):
    gallery_id: str
    post_id: str
    title: Optional[str]
    state: str


class StatusResponse(BaseModel):
    gallery_id: str
    name: Optional[str]
    running: bool
    total_posts: int
    last_activity: Optional[datetime]
    last_error: Optional[str]
    queue_size: int
    queue: list[QueueItemResponse]
    mini: bool
    delay: float
    fetch_delay: float
    pages: int
    followup_delays: list[float]
    auto_start: bool
    max_comment_pages: int
    comment_page_size: int
    comment_delay: float
    comment_followup_delay: float
    comment_followup_threshold: float
    comment_followup_max_retries: int
    db_size_bytes: int


class PostResponse(BaseModel):
    post_id: str
    title: Optional[str]
    nickname: Optional[str]
    user_id: Optional[str]
    created_at: Optional[str]
    view_count: Optional[int]
    recommend_count: Optional[int]
    dislike_count: Optional[int]
    is_deleted: bool


class PageResponse(BaseModel):
    items: list[PostResponse]
    page: int
    page_size: int
    total: int
    total_pages: int


class DashboardGalleryPosts(BaseModel):
    gallery_id: str
    name: Optional[str]
    posts: list[PostResponse]


class QueueOverviewItemResponse(BaseModel):
    queue_id: int
    gallery_id: str
    post_id: str
    action: str
    state: str
    manual: bool
    available_at: float
    available_in: float


class QueueOverviewResponse(BaseModel):
    collecting_total: int
    collecting: list[QueueOverviewItemResponse]
    scheduled_total: int
    scheduled: list[QueueOverviewItemResponse]
    generated_at: datetime


class ConfigRawResponse(BaseModel):
    content: str


class ConfigRawRequest(BaseModel):
    content: str


class SummaryNotablePost(BaseModel):
    post_id: Optional[str] = None
    title: Optional[str] = None
    reason: Optional[str] = None


class SummaryRequest(BaseModel):
    search: Optional[str] = None
    author: Optional[str] = None
    user_id: Optional[str] = Field(default=None, alias="user_id")
    deleted_only: bool = False

    model_config = ConfigDict(populate_by_name=True)


class SummaryResponseModel(BaseModel):
    post_count: int
    summary: Optional[str] = None
    tone: Optional[str] = None
    common_topics: list[str] = Field(default_factory=list)
    notable_posts: list[SummaryNotablePost] = Field(default_factory=list)
    raw: Optional[str] = None


class CreateGalleryRequest(BaseModel):
    id: str = Field(..., description="DCInside gallery identifier")
    name: Optional[str] = Field(default=None, description="Friendly name")
    mini: bool = Field(default=False, description="Use mini gallery endpoints")
    delay: float = Field(default=30.0, ge=1.0)
    fetch_delay: float = Field(default=1.0, ge=0.0)
    pages: int = Field(default=1, ge=1, le=10)
    followup_delays: list[float] = Field(
        default_factory=lambda: [60.0, 300.0, 1800.0]
    )
    auto_start: bool = True
    max_comment_pages: int = Field(default=5, ge=1)
    comment_page_size: int = Field(default=100, ge=10)
    comment_delay: float = Field(default=0.5, ge=0.0)
    comment_followup_delay: float = Field(default=1800.0, ge=0.0)
    comment_followup_threshold: float = Field(default=300.0, ge=0.0)
    comment_followup_max_retries: int = Field(default=3, ge=0)


class UpdateGalleryRequest(BaseModel):
    name: Optional[str] = Field(default=None)
    mini: Optional[bool] = Field(default=None)
    delay: Optional[float] = Field(default=None, ge=1.0)
    fetch_delay: Optional[float] = Field(default=None, ge=0.0)
    pages: Optional[int] = Field(default=None, ge=1, le=10)
    followup_delays: Optional[list[float]] = None
    auto_start: Optional[bool] = None
    max_comment_pages: Optional[int] = Field(default=None, ge=1)
    comment_page_size: Optional[int] = Field(default=None, ge=10)
    comment_delay: Optional[float] = Field(default=None, ge=0.0)
    comment_followup_delay: Optional[float] = Field(default=None, ge=0.0)
    comment_followup_threshold: Optional[float] = Field(default=None, ge=0.0)
    comment_followup_max_retries: Optional[int] = Field(default=None, ge=0)


def create_app() -> FastAPI:
    config = ArchiveConfig.load()
    manager = ArchiveManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = manager
        await manager.start_auto()
        try:
            yield
        finally:
            await manager.close()

    app = FastAPI(lifespan=lifespan, title="DCInside Archiver API")
    app.state.manager = manager

    # Allow dashboard JavaScript to make API calls.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"]
,
        allow_headers=["*"],
    )

    static_dir = DASHBOARD_PATH.parent
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def get_manager(request: Request) -> ArchiveManager:
        manager = getattr(request.app.state, "manager", None)
        if manager is None:
            raise HTTPException(status_code=503, detail="Archive manager not ready")
        return manager

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return DASHBOARD_PATH.read_text("utf-8")

    @app.get("/view.html", response_class=HTMLResponse)
    async def view_page() -> str:
        return VIEW_PATH.read_text("utf-8")

    @app.get("/list.html", response_class=HTMLResponse)
    async def list_page() -> str:
        return LIST_PATH.read_text("utf-8")

    @app.get("/api/status", response_model=list[StatusResponse])
    async def get_status(manager: ArchiveManager = Depends(get_manager)) -> list[StatusResponse]:
        statuses = await manager.status_snapshot()
        return [_status_to_response(status) for status in statuses]

    @app.post("/api/galleries", response_model=StatusResponse, status_code=201)
    async def create_gallery(
        payload: CreateGalleryRequest,
        manager: ArchiveManager = Depends(get_manager),
    ) -> StatusResponse:
        try:
            settings = GallerySettings(**payload.model_dump())
            status = await manager.add_gallery(settings)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _status_to_response(status)

    @app.patch("/api/galleries/{gallery_id}", response_model=StatusResponse)
    async def update_gallery(
        gallery_id: str,
        payload: UpdateGalleryRequest,
        manager: ArchiveManager = Depends(get_manager),
    ) -> StatusResponse:
        changes = payload.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(status_code=400, detail="No changes provided")
        try:
            status = await manager.update_gallery(gallery_id, **changes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Gallery not found") from exc
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _status_to_response(status)

    @app.post("/api/galleries/{gallery_id}/start")
    async def start_gallery(
        gallery_id: str, manager: ArchiveManager = Depends(get_manager)
    ) -> dict[str, str]:
        archiver = manager.get_archiver(gallery_id)
        if not archiver:
            raise HTTPException(status_code=404, detail="Gallery not found")
        await archiver.start()
        return {"status": "started"}

    @app.post("/api/galleries/{gallery_id}/stop")
    async def stop_gallery(
        gallery_id: str, manager: ArchiveManager = Depends(get_manager)
    ) -> dict[str, str]:
        archiver = manager.get_archiver(gallery_id)
        if not archiver:
            raise HTTPException(status_code=404, detail="Gallery not found")
        await archiver.stop()
        return {"status": "stopped"}

    @app.delete("/api/galleries/{gallery_id}")
    async def delete_gallery(
        gallery_id: str, manager: ArchiveManager = Depends(get_manager)
    ) -> dict[str, bool]:
        try:
            await manager.delete_gallery(gallery_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Gallery not found") from exc
        return {"ok": True}

    @app.get("/api/galleries/{gallery_id}/posts", response_model=PageResponse)
    async def list_posts(
        gallery_id: str,
        manager: ArchiveManager = Depends(get_manager),
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        author: Optional[str] = None,
        user_id: Optional[str] = None,
        deleted_only: bool = Query(False, alias="deleted_only"),
    ) -> PageResponse:
        try:
            if search:
                page_data = await manager.search_posts_paginated(
                    gallery_id,
                    search,
                    page=page,
                    page_size=page_size,
                    deleted_only=deleted_only,
                )
            elif author:
                page_data = await manager.search_posts_by_author(
                    gallery_id,
                    author,
                    page=page,
                    page_size=page_size,
                    deleted_only=deleted_only,
                )
            elif user_id:
                page_data = await manager.search_posts_by_user_id(
                    gallery_id,
                    user_id,
                    page=page,
                    page_size=page_size,
                    deleted_only=deleted_only,
                )
            else:
                page_data = await manager.list_posts(
                    gallery_id,
                    page=page,
                    page_size=page_size,
                    deleted_only=deleted_only,
                )
        except KeyError:
            raise HTTPException(status_code=404, detail="Gallery not found")
        return _page_to_response(page_data)

    @app.post("/api/config/reload")
    async def reload_config(manager: ArchiveManager = Depends(get_manager)) -> dict[str, str]:
        await manager.reload()
        return {"status": "reloaded"}

    @app.get("/api/config/raw", response_model=ConfigRawResponse)
    async def get_config_raw(manager: ArchiveManager = Depends(get_manager)) -> ConfigRawResponse:
        path = manager.config.path
        if path.exists():
            content = path.read_text("utf-8")
        else:
            content = ""
        return ConfigRawResponse(content=content)

    @app.put("/api/config/raw")
    async def update_config_raw(
        payload: ConfigRawRequest, manager: ArchiveManager = Depends(get_manager)
    ) -> dict[str, str]:
        try:
            data = tomllib.loads(payload.content)
            ArchiveConfigModel(**data)
        except (tomllib.TOMLDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc

        path = manager.config.path
        path.write_text(payload.content, encoding="utf-8")
        await manager.reload()
        return {"status": "saved"}

    @app.get("/api/galleries/{gallery_id}/posts/{post_id}")
    async def fetch_post(
        gallery_id: str,
        post_id: str,
        manager: ArchiveManager = Depends(get_manager),
    ) -> dict:
        try:
            post = await manager.fetch_post(gallery_id, post_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Gallery not found")
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post

    @app.post("/api/galleries/{gallery_id}/posts/{post_id}/recrawl")
    async def recrawl_post(
        gallery_id: str,
        post_id: str,
        manager: ArchiveManager = Depends(get_manager),
    ) -> dict[str, str]:
        try:
            await manager.recrawl_post(gallery_id, post_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Gallery not found") from exc
        return {"status": "refreshed"}

    @app.get("/api/dashboard/posts", response_model=list[DashboardGalleryPosts])
    async def dashboard_posts(
        limit: int = 5, manager: ArchiveManager = Depends(get_manager)
    ) -> list[DashboardGalleryPosts]:
        galleries: list[DashboardGalleryPosts] = []
        for archiver in manager.iter_archivers():
            try:
                page = await manager.list_posts(
                    archiver.settings.id, page=1, page_size=limit
                )
            except KeyError:
                continue
            galleries.append(
                DashboardGalleryPosts(
                    gallery_id=archiver.settings.id,
                    name=archiver.settings.name,
                    posts=[_post_to_response(item) for item in page.items],
                )
            )
        return galleries

    @app.get("/api/dashboard/queue", response_model=QueueOverviewResponse)
    async def dashboard_queue(
        manager: ArchiveManager = Depends(get_manager),
        limit: int = Query(10, ge=1, le=50),
    ) -> QueueOverviewResponse:
        overview = await manager.queue_overview(limit=limit)
        now = time.time()
        collecting = [
            QueueOverviewItemResponse(
                queue_id=item.queue_id,
                gallery_id=item.gallery_id,
                post_id=item.post_id,
                action=item.action,
                state=item.state,
                manual=item.manual,
                available_at=item.available_at,
                available_in=max(0.0, item.available_at - now),
            )
            for item in overview.collecting
        ]
        scheduled = [
            QueueOverviewItemResponse(
                queue_id=item.queue_id,
                gallery_id=item.gallery_id,
                post_id=item.post_id,
                action=item.action,
                state=item.state,
                manual=item.manual,
                available_at=item.available_at,
                available_in=max(0.0, item.available_at - now),
            )
            for item in overview.scheduled
        ]
        return QueueOverviewResponse(
            collecting_total=overview.collecting_total,
            collecting=collecting,
            scheduled_total=overview.scheduled_total,
            scheduled=scheduled,
            generated_at=datetime.now(timezone.utc),
        )

    @app.post(
        "/api/galleries/{gallery_id}/llm-summary",
        response_model=SummaryResponseModel,
    )
    async def summarize_gallery_posts(
        gallery_id: str,
        payload: SummaryRequest,
        manager: ArchiveManager = Depends(get_manager),
    ) -> SummaryResponseModel:
        try:
            rows = await manager.fetch_posts_for_summary(
                gallery_id,
                limit=50,
                search=payload.search,
                author=payload.author,
                user_id=payload.user_id,
                deleted_only=payload.deleted_only,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Gallery not found")

        llm_config = manager.config.model.llm
        api_key = (llm_config.api_key or "").strip()
        model_name = (llm_config.model or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="LLM API 키가 설정되어 있지 않습니다.")
        if not model_name:
            raise HTTPException(status_code=400, detail="LLM 모델을 설정해주세요.")

        posts: list[LLMPost] = []
        for row in rows:
            post_id = str(row.get("post_id")) if row.get("post_id") is not None else ""
            title = str(row.get("title") or "").strip()
            author = str(
                row.get("nickname")
                or row.get("user_id")
                or "익명"
            ).strip()
            created_at = row.get("created_at")
            content = str(row.get("content") or "")
            posts.append(
                LLMPost(
                    post_id=post_id,
                    title=title or "제목 없음",
                    author=author or "익명",
                    created_at=created_at,
                    content=content,
                )
            )

        if not posts:
            return SummaryResponseModel(post_count=0)

        service = LLMService(
            api_key=api_key,
            model=model_name,
            prompt=(llm_config.summary_prompt or "").strip() or None,
        )
        try:
            result = await service.summarize_posts(posts)
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return SummaryResponseModel(
            post_count=len(posts),
            summary=result.summary,
            tone=result.tone,
            common_topics=list(result.common_topics),
            notable_posts=[
                SummaryNotablePost(
                    post_id=item.post_id,
                    title=item.title,
                    reason=item.reason,
                )
                for item in result.notable_posts
            ],
            raw=result.raw,
        )

    return app


def _status_to_response(status: GalleryStatus) -> StatusResponse:
    queue_items = [
        QueueItemResponse(
            gallery_id=item.gallery_id,
            post_id=item.post_id,
            title=item.title,
            state=item.state,
        )
        for item in status.queue
    ]
    return StatusResponse(
        gallery_id=status.gallery_id,
        name=status.name,
        running=status.running,
        total_posts=status.total_posts,
        last_activity=status.last_activity,
        last_error=status.last_error,
        queue_size=status.queue_size,
        queue=queue_items,
        mini=status.mini,
        delay=status.delay,
        fetch_delay=status.fetch_delay,
        pages=status.pages,
        followup_delays=list(status.followup_delays),
        auto_start=status.auto_start,
        max_comment_pages=status.max_comment_pages,
        comment_page_size=status.comment_page_size,
        comment_delay=status.comment_delay,
        comment_followup_delay=status.comment_followup_delay,
        comment_followup_threshold=status.comment_followup_threshold,
        comment_followup_max_retries=status.comment_followup_max_retries,
        db_size_bytes=status.db_size_bytes,
    )


def _page_to_response(page: PostPage) -> PageResponse:
    items = [_post_to_response(item) for item in page.items]
    return PageResponse(
        items=items,
        page=page.page,
        page_size=page.page_size,
        total=page.total,
        total_pages=page.total_pages,
    )


def _post_to_response(item: PostListItem) -> PostResponse:
    return PostResponse(
        post_id=item.post_id,
        title=item.title,
        nickname=item.nickname,
        user_id=item.user_id,
        created_at=item.created_at,
        view_count=item.view_count,
        recommend_count=item.recommend_count,
        dislike_count=item.dislike_count,
        is_deleted=item.is_deleted,
    )


app = create_app()


async def main() -> None:
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
