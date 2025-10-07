"""Asynchronous helpers for scraping DCInside using the mobile site."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag
from urllib.parse import quote

LOGGER = logging.getLogger(__name__)


MOBILE_BASE_URL = "https://m.dcinside.com"
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 "
    "Mobile/15E148 Safari/604.1"
)
MOBILE_HEADERS = {
    "User-Agent": MOBILE_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _looks_like_mobile_user_agent(user_agent: str) -> bool:
    """Return ``True`` when *user_agent* appears to describe a mobile browser."""

    lowered = user_agent.lower()
    return any(token in lowered for token in ("mobile", "iphone", "ipad", "android", "ipod"))
KST = timezone(timedelta(hours=9))


@dataclass(slots=True)
class Author:
    nickname: str
    user_id: Optional[str] = None
    ip: Optional[str] = None


@dataclass(slots=True)
class PostSummary:
    gallery_id: str
    post_id: str
    type: str
    subject: str
    title: str
    link: str
    author: Author
    created_at: str
    view_count: int
    recommend_count: int
    comment_count: int


@dataclass(slots=True)
class Comment:
    comment_id: str
    parent_id: Optional[str]
    author: Author
    created_at: str
    content: str
    dccons: Sequence[str] = field(default_factory=tuple)
    is_deleted: bool = False

    @property
    def images(self) -> Sequence[str]:  # pragma: no cover - compatibility shim
        """Alias for backwards compatibility."""
        return self.dccons


@dataclass(slots=True)
class PostDetail:
    gallery_id: str
    post_id: str
    title: str
    author: Author
    created_at: str
    content: str
    images: Sequence[str] = field(default_factory=tuple)
    view_count: int = 0
    recommend_count: int = 0
    dislike_count: int = 0
    comments: Sequence[Comment] = field(default_factory=tuple)
    is_deleted: bool = False
    raw: Optional[str] = None


def _absolute_url(base_url: str, url: str | None) -> str:
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    prefix = base_url.rstrip("/")
    suffix = url if url.startswith("/") else f"/{url}"
    return f"{prefix}{suffix}"


def _extract_number(value: str) -> int:
    match = re.search(r"([\d,]+)", value or "")
    if not match:
        return 0
    return int(match.group(1).replace(",", ""))


def _sanitize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    sanitized = "\n".join(lines).strip()
    return sanitized


def _normalize_board_datetime(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    now = datetime.now(KST)
    if re.fullmatch(r"\d{2}:\d{2}", raw):
        hour, minute = map(int, raw.split(":"))
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt.isoformat()
    if re.fullmatch(r"\d{2}\.\d{2}", raw):
        month, day = map(int, raw.split("."))
        dt = datetime(now.year, month, day, tzinfo=KST)
        return dt.isoformat()
    if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}", raw):
        dt = datetime.strptime(raw, "%Y.%m.%d %H:%M").replace(tzinfo=KST)
        return dt.isoformat()
    return raw


def _normalize_comment_datetime(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    now = datetime.now(KST)
    if re.fullmatch(r"\d{2}\.\d{2}\s+\d{2}:\d{2}", raw):
        month, day = map(int, raw[:5].split("."))
        hour, minute = map(int, raw[-5:].split(":"))
        dt = datetime(now.year, month, day, hour, minute, tzinfo=KST)
        return dt.isoformat()
    if re.fullmatch(r"\d{2}:\d{2}", raw):
        hour, minute = map(int, raw.split(":"))
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt.isoformat()
    return raw


def _split_author(value: str) -> tuple[str, Optional[str]]:
    text = (value or "").strip()
    if not text:
        return "익명", None
    match = re.match(r"(.+?)\(([^)]+)\)$", text)
    if not match:
        return text, None
    ip = match.group(2).strip()
    nickname = text or "익명"
    return nickname, ip


class PostDeletedError(RuntimeError):
    """Raised when a post can no longer be retrieved from DCInside."""

    def __init__(self, gallery_id: str, post_id: str, status_code: int) -> None:
        self.gallery_id = gallery_id
        self.post_id = post_id
        self.status_code = status_code
        super().__init__(
            f"Post {gallery_id}/{post_id} unavailable (HTTP {status_code})"
        )


class RateLimitedError(RuntimeError):
    """Raised when DCInside responds with HTTP 429."""

    def __init__(self, retry_after: float = 60.0) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited by DCInside; retry after {retry_after:.0f}s")


class DCInsideClient:
    """Asynchronous scraper that favours the mobile DCInside views."""

    def __init__(
        self,
        *,
        base_url: str = MOBILE_BASE_URL,
        user_agent: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") or MOBILE_BASE_URL
        headers = dict(MOBILE_HEADERS)
        if user_agent:
            if _looks_like_mobile_user_agent(user_agent):
                headers["User-Agent"] = user_agent
            else:
                LOGGER.warning(
                    "User agent %s does not look like a mobile browser; using default mobile headers",
                    user_agent,
                )
        # DCInside serves different templates based on the negotiated protocol. Using
        # HTTP/2 causes the mobile endpoints to redirect to the desktop site which the
        # parser cannot handle. Stick to HTTP/1.1 unless a custom client is supplied.
        self._client = client or httpx.AsyncClient(headers=headers, timeout=20.0)
        self._owns_client = client is None
        self._html_headers = dict(self._client.headers)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_board_page(
        self,
        gallery_id: str,
        *,
        page: int = 1,
        board_type: str = "all",
    ) -> list[PostSummary]:
        if page <= 0:
            return []

        query = "recommend=1" if board_type == "recommend" else f"page={page}"
        url = f"{self.base_url}/board/{quote(gallery_id, safe='')}?{query}"
        response = await self._client.get(url, headers={"Cookie": "list_count=100"})
        if response.status_code == 429:
            LOGGER.warning("Rate limited while fetching board %s page %s", gallery_id, page)
            raise RateLimitedError()
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        posts: list[PostSummary] = []

        for row in soup.select("ul.gall-detail-lst > li"):
            if row.select_one(".pwlink") or row.select_one(".power-lst"):
                continue

            classes = set(row.get("class") or [])
            if board_type == "notice" and "notice" not in classes:
                continue
            if board_type != "notice" and "notice" in classes:
                continue

            anchor = row.select_one("a")
            if not anchor:
                continue
            href = anchor.get("href") or ""
            match = re.search(r"/(\d+)(?:\?|$)", href)
            if not match:
                continue
            post_id = match.group(1)

            type_map = {
                "sp-lst-txt": "text",
                "sp-lst-img": "picture",
                "sp-lst-recoimg": "recommended",
                "sp-lst-recotxt": "recommended",
            }
            icon = row.select_one(".subject-add .sp-lst")
            icon_key = None
            if icon and icon.has_attr("class"):
                icon_key = next((cls for cls in icon["class"] if cls.startswith("sp-lst-")), None)
            post_type = type_map.get(icon_key or "", "unknown")

            subject_el = row.select_one(".ginfo > li:nth-of-type(1)")
            date_el = row.select_one(".ginfo > li:nth-of-type(2)")
            view_el = row.select_one(".ginfo > li:nth-of-type(3)")
            recommend_el = row.select_one(".ginfo > li:nth-of-type(4)")

            subject = subject_el.get_text(strip=True) if subject_el else ""
            created_at = _normalize_board_datetime(date_el.get_text(strip=True) if date_el else "")
            view_count = _extract_number(view_el.get_text(strip=True) if view_el else "0")
            recommend_count = _extract_number(recommend_el.get_text(strip=True) if recommend_el else "0")

            block = row.select_one(".blockInfo")
            nickname = block.get("data-name") if block and block.get("data-name") else "익명"
            data_info = block.get("data-info") if block else None
            user_id: Optional[str] = None
            ip: Optional[str] = None
            if data_info:
                if "." in data_info:
                    ip = data_info
                else:
                    user_id = data_info
            if ip and nickname and not nickname.endswith(f"({ip})"):
                nickname = f"{nickname}({ip})"

            comments_el = row.select_one(".ct")
            comment_count = _extract_number(comments_el.get_text(strip=True) if comments_el else "0")

            title = row.select_one(".subjectin")
            title_text = " ".join((title.get_text() if title else "").split())

            posts.append(
                PostSummary(
                    gallery_id=gallery_id,
                    post_id=post_id,
                    type=post_type,
                    subject=subject,
                    title=title_text,
                    link=_absolute_url(self.base_url, href),
                    author=Author(nickname=nickname, user_id=user_id, ip=ip),
                    created_at=created_at,
                    view_count=view_count,
                    recommend_count=recommend_count,
                    comment_count=comment_count,
                )
            )

        return posts

    async def download_image(self, url: str) -> tuple[bytes, Optional[str]]:
        """Retrieve an image and return its binary content and content type."""

        if not url:
            raise ValueError("Image URL must not be empty")

        absolute_url = _absolute_url(self.base_url, url)
        response = await self._client.get(
            absolute_url, headers={"Referer": self.base_url}
        )
        if response.status_code == 429:
            LOGGER.warning("Rate limited while downloading image %s", absolute_url)
            raise RateLimitedError()
        response.raise_for_status()
        return response.content, response.headers.get("Content-Type")

    async def fetch_post(
        self,
        gallery_id: str,
        post_id: str | int,
        *,
        extract_images: bool = True,
        include_image_source: bool = False,
        fetch_comments: bool = True,
        max_comment_pages: int = 1,
        comment_page_size: int = 100,
        comment_delay: float = 0.0,
    ) -> Optional[PostDetail]:
        del max_comment_pages, comment_page_size, comment_delay
        html = await self._get_mobile_post_html(gallery_id, post_id)
        soup = BeautifulSoup(html, "html.parser")
        detail = self._parse_mobile_post(
            soup,
            gallery_id,
            str(post_id),
            extract_images=extract_images,
            include_image_source=include_image_source,
            include_comments=fetch_comments,
        )
        detail.raw = html
        return detail

    async def fetch_comments(
        self,
        gallery_id: str,
        post_id: str | int,
        *,
        max_pages: int = 1,
        page_size: int = 100,
        delay: float = 0.0,
    ) -> Sequence[Comment]:
        del max_pages, page_size, delay
        html = await self._get_mobile_post_html(gallery_id, post_id)
        soup = BeautifulSoup(html, "html.parser")
        return self._parse_mobile_comments(soup)

    async def _get_mobile_post_html(self, gallery_id: str, post_id: str | int) -> str:
        post_no = str(post_id)
        if not post_no:
            raise ValueError("post_id is required")
        url = f"{self.base_url}/board/{quote(gallery_id, safe='')}/{quote(post_no, safe='')}"
        response = await self._client.get(url, headers=self._html_headers)
        if response.status_code == 429:
            LOGGER.warning("Rate limited while fetching post %s/%s", gallery_id, post_no)
            raise RateLimitedError()
        if response.status_code in {403, 404, 410, 451}:
            LOGGER.info(
                "Post %s/%s returned status %s", gallery_id, post_no, response.status_code
            )
            raise PostDeletedError(gallery_id, post_no, response.status_code)
        response.raise_for_status()
        return response.text

    def _parse_mobile_post(
        self,
        soup: BeautifulSoup,
        gallery_id: str,
        post_id: str,
        *,
        extract_images: bool,
        include_image_source: bool,
        include_comments: bool,
    ) -> PostDetail:
        title_el = soup.select_one("span.tit") or soup.select_one("meta[property='og:title']")
        title_text = ""
        if title_el:
            title_text = (
                title_el.get("content") if title_el.name == "meta" else title_el.get_text(strip=True)
            )
            if title_text:
                title_text = " ".join(title_text.split())
        infos = soup.select("ul.ginfo2 li")
        author_text = ""
        user_id: Optional[str] = None
        if infos:
            author_item = infos[0]
            author_link = author_item.select_one("a[href]")
            if author_link:
                for span in author_link.find_all("span"):
                    span.extract()
                author_text = author_link.get_text(strip=True)
                href = author_link.get("href") or ""
                match = re.search(r"/gallog/([^/?#]+)", href)
                if match:
                    user_id = match.group(1)
            else:
                author_text = author_item.get_text(strip=True)
        nickname, ip = _split_author(author_text)
        created_at = ""
        view_count = recommend_count = dislike_count = 0
        for item in infos[1:]:
            text = item.get_text(strip=True)
            if not created_at and re.search(r"\d{2}:\d{2}", text):
                created_at = _normalize_board_datetime(text)
            elif "조회" in text:
                view_count = _extract_number(text)
            elif "추천" in text:
                recommend_count = _extract_number(text)
        if not created_at:
            date_meta = soup.select_one("meta[property='og:regDate']")
            if date_meta and date_meta.get("content"):
                created_at = date_meta["content"].strip()
        if not nickname:
            nickname = "익명"
        author = Author(nickname=nickname, user_id=user_id, ip=ip)

        content_el = soup.select_one(".thum-txtin")
        is_deleted = False
        images: list[str] = []
        if content_el is None:
            placeholder = soup.select_one(".thum-txtin p") or soup.select_one(".result")
            content_text = placeholder.get_text(strip=True) if placeholder else ""
            is_deleted = True
        else:
            for br in content_el.find_all("br"):
                br.replace_with("\n")
            for block in content_el.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
                block.insert_after("\n")
            if extract_images:
                images = self._process_images(content_el, include_source=include_image_source)
            else:
                self._process_images(content_el, include_source=False, replace_only=True)
            content_text = _sanitize_text(content_el.get_text())

        recommend_btn = soup.select_one("#recomm_btn")
        if recommend_btn:
            recommend_count = _extract_number(recommend_btn.get_text(strip=True)) or recommend_count
        dislike_btn = soup.select_one("#nonrecomm_btn")
        if dislike_btn:
            dislike_count = _extract_number(dislike_btn.get_text(strip=True))

        comments: Sequence[Comment] = ()
        if include_comments:
            comments = self._parse_mobile_comments(soup)

        return PostDetail(
            gallery_id=gallery_id,
            post_id=post_id,
            title=title_text,
            author=author,
            created_at=created_at,
            content=content_text,
            images=tuple(images),
            view_count=view_count,
            recommend_count=recommend_count,
            dislike_count=dislike_count,
            comments=tuple(comments),
            is_deleted=is_deleted,
        )

    def _resolve_image_url(self, tag: Tag) -> str:
        url = (
            tag.get("data-original")
            or tag.get("data-src")
            or tag.get("src")
            or ""
        )
        if not url:
            return ""
        if not url.startswith("http"):
            url = _absolute_url(self.base_url, url)
        return url

    def _process_images(
        self,
        container: Tag,
        *,
        include_source: bool,
        replace_only: bool = False,
        start_index: int = 0,
        label: str = "image",
    ) -> list[str]:
        images: list[str] = []
        for img in container.find_all("img"):
            url = self._resolve_image_url(img)
            if not url:
                continue
            index = start_index + len(images)
            images.append(url)
            placeholder = f"[{label}({index})]"
            if include_source:
                placeholder = f"[{label}({index}): {url}]"
            if replace_only:
                img.replace_with(placeholder)
            else:
                img.replace_with(placeholder + "\n")
        return images

    def _parse_mobile_comments(self, soup: BeautifulSoup) -> Sequence[Comment]:
        comments: list[Comment] = []
        current_parent: Optional[str] = None
        for item in soup.select(".all-comment-lst > li"):
            comment_id = (item.get("no") or "").strip()
            if not comment_id:
                continue
            classes = item.get("class") or []
            author_link = item.select_one("a.nick")
            nickname_text = author_link.get_text(strip=True) if author_link else "익명"
            if author_link:
                for span in author_link.find_all("span"):
                    span.extract()
            gallog_id: Optional[str] = None
            if author_link and author_link.get("href"):
                href = author_link["href"]
                match = re.search(r"/gallog/([^/?#]+)", href)
                if match:
                    gallog_id = match.group(1)
            block_id = item.select_one(".blockCommentId")
            data_info = block_id.get("data-info") if block_id else None
            user_id: Optional[str] = None
            ip: Optional[str] = None
            ip_span = item.select_one(".ip")
            if gallog_id:
                user_id = gallog_id
            if data_info and "." not in data_info and not user_id:
                user_id = data_info
            if ip_span and ip_span.get_text(strip=True):
                ip = ip_span.get_text(strip=True).strip("()")
            elif data_info and "." in data_info:
                ip = data_info
            nickname = nickname_text.strip() or "익명"

            content_el = item.select_one("p.txt")
            comment_dccons: list[str] = []
            text_parts: list[str] = []
            if content_el:
                for br in content_el.find_all("br"):
                    br.replace_with("\n")
                comment_dccons.extend(
                    self._process_images(
                        content_el,
                        include_source=True,
                        start_index=0,
                        label="dccon",
                    )
                )
                text_parts.append(content_el.get_text())

            attachment_placeholders: list[str] = []
            for extra in item.select(".comment-add-img, .re-img, .comment-add-file"):
                for img in extra.find_all("img"):
                    url = self._resolve_image_url(img)
                    if not url:
                        continue
                    index = len(comment_dccons)
                    comment_dccons.append(url)
                    attachment_placeholders.append(f"[dccon({index}): {url}]")

            if attachment_placeholders:
                text_parts.extend(attachment_placeholders)

            content_text = _sanitize_text("\n".join(text_parts))

            date_el = item.select_one("span.date")
            created_at = _normalize_comment_datetime(date_el.get_text(strip=True) if date_el else "")

            is_deleted = False
            if any(cls in {"comment-blind", "comment-block", "del"} for cls in classes):
                is_deleted = True
            if not content_text and "삭제" in item.get_text(strip=True):
                is_deleted = True

            parent_attr = (
                item.get("parent-no")
                or item.get("parent_no")
                or item.get("data-parent-no")
                or item.get("data-parent")
            )
            if parent_attr:
                parent_id = parent_attr.strip() or "0"
            elif any(cls in {"comment-add", "re", "reply", "cmt-reply", "depth2"} for cls in classes):
                parent_id = current_parent or "0"
            else:
                current_parent = comment_id
                parent_id = "0"

            if parent_id == "0":
                current_parent = comment_id

            comments.append(
                Comment(
                    comment_id=comment_id,
                    parent_id=parent_id,
                    author=Author(nickname=nickname, user_id=user_id, ip=ip),
                    created_at=created_at,
                    content=content_text,
                    dccons=tuple(comment_dccons),
                    is_deleted=is_deleted,
                )
            )
        return comments


__all__ = [
    "Author",
    "Comment",
    "PostDeletedError",
    "RateLimitedError",
    "PostDetail",
    "PostSummary",
    "DCInsideClient",
]
