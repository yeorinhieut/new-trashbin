"""Utilities for generating LLM-based summaries of gallery posts."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Sequence


try:  # pragma: no cover - optional dependency guard
    from openai import OpenAI
except Exception:  # pragma: no cover - import guard for missing dependency
    OpenAI = None  # type: ignore[assignment]


DEFAULT_SUMMARY_PROMPT = (
    "다음은 최신 게시글 목록입니다. 전체 분위기와 핵심 이슈를 한 문장으로 요약하세요. "
    "출력은 한국어 평문으로 80자 이내 한 문장으로만 작성합니다."
)


__all__ = [
    "DEFAULT_SUMMARY_PROMPT",
    "LLMError",
    "LLMConfigurationError",
    "LLMPost",
    "NotablePost",
    "LLMSummary",
    "LLMService",
]


class LLMError(RuntimeError):
    """Base error for LLM operations."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM is not correctly configured."""


@dataclass(slots=True)
class LLMPost:
    post_id: str
    title: str
    author: str
    created_at: str | None
    content: str


@dataclass(slots=True)
class NotablePost:
    post_id: str | None = None
    title: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class LLMSummary:
    summary: str | None = None
    tone: str | None = None
    common_topics: list[str] = field(default_factory=list)
    notable_posts: list[NotablePost] = field(default_factory=list)
    raw: str | None = None


class LLMService:
    """Wrapper around the OpenAI client for generating gallery summaries."""

    def __init__(self, api_key: str, model: str, prompt: str | None = None) -> None:
        if not api_key:
            raise LLMConfigurationError("OpenAI API 키가 설정되어 있지 않습니다.")
        if not model:
            raise LLMConfigurationError("사용할 모델을 설정해주세요.")
        if OpenAI is None:  # pragma: no cover - optional dependency
            raise LLMConfigurationError("openai 패키지가 설치되어 있지 않습니다.")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._prompt_template = (prompt or DEFAULT_SUMMARY_PROMPT).strip() or DEFAULT_SUMMARY_PROMPT

    async def summarize_posts(self, posts: Sequence[LLMPost]) -> LLMSummary:
        if not posts:
            return LLMSummary()

        prompt = self._build_prompt(posts)

        def _call_openai() -> Any:
            input_payload = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You analyse DCInside gallery posts and summarise them "
                                "in Korean. Provide a single concise sentence without "
                                "markdown or bullet points."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ]

            request: dict[str, Any] = {
                "model": self._model,
                "max_output_tokens": 1200,
                "input": input_payload,
            }

            return self._client.responses.create(**request)

        try:
            response = await asyncio.to_thread(_call_openai)
        except Exception as exc:  # pragma: no cover - network/path errors
            raise LLMError(f"LLM 요청에 실패했습니다: {exc}") from exc

        raw_text = _extract_output_text(response).strip()
        summary = LLMSummary(raw=raw_text or None)

        if not raw_text:
            return summary

        summary.summary = _summarise_single_sentence(raw_text)
        return summary

    def _build_prompt(self, posts: Sequence[LLMPost]) -> str:
        sections: list[str] = [self._prompt_template]
        for index, post in enumerate(posts, start=1):
            lines = [f"{index}. 글번호: {post.post_id}"]
            lines.append(f"제목: {post.title}")
            lines.append(f"작성자: {post.author}")
            if post.created_at:
                lines.append(f"작성일: {post.created_at}")
            content = post.content.strip()
            lines.append("본문:")
            lines.append(content if content else "(본문 없음)")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        fragments: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for segment in content:
                    if isinstance(segment, dict):
                        value = segment.get("text")
                        if isinstance(value, str):
                            fragments.append(value)
                        elif "json" in segment:
                            try:
                                fragments.append(json.dumps(segment["json"], ensure_ascii=False))
                            except Exception:  # pragma: no cover - best effort fallback
                                continue
                    else:
                        value = getattr(segment, "text", None)
                        if isinstance(value, str):
                            fragments.append(value)
        if fragments:
            return "".join(fragments)
    return ""


def _summarise_single_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    first_line = cleaned.splitlines()[0].strip()
    if not first_line:
        return cleaned

    # If the model prepends a label such as "요약:" remove it for display purposes.
    if ":" in first_line:
        label, _, remainder = first_line.partition(":")
        if len(label) <= 10 and remainder.strip():
            first_line = remainder.strip()

    return first_line
