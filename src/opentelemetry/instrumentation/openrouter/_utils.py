"""Shared helpers for the OpenRouter wrappers."""

from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar
from urllib.parse import urlsplit

from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv.attributes.server_attributes import (
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from opentelemetry.instrumentation.openrouter import _attributes as attrs

logger = logging.getLogger(__name__)

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)

F = TypeVar("F", bound=Callable[..., Any])


def dont_throw(func: F) -> F:
    """Run ``func`` and log (at debug level) any exception it raises.

    Instrumentation code must never break the instrumented SDK call.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug(
                "OpenRouter instrumentation failed in %s",
                func.__qualname__,
                exc_info=True,
            )
            return None

    return wrapper  # type: ignore[return-value]


def is_content_capture_enabled(override: bool | None) -> bool:
    if override is not None:
        return override
    value = os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "")
    return value.strip().lower() == "true"


def get_field(obj: Any, *names: str) -> Any:
    """Read the first present field from a pydantic model or a dict."""
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def set_attr(span: Span, key: str, value: Any) -> None:
    if value is None:
        return
    span.set_attribute(key, value)


@dont_throw
def set_server_attributes(span: Span, instance: Any, kwargs: dict[str, Any]) -> None:
    url = kwargs.get("server_url")
    if not url:
        url, _ = instance.sdk_configuration.get_server_details()
    if not url:
        return
    parts = urlsplit(url)
    if not parts.hostname:
        return
    span.set_attribute(SERVER_ADDRESS, parts.hostname)
    port = parts.port
    if port is None:
        port = {"https": 443, "http": 80}.get(parts.scheme)
    if port is not None:
        span.set_attribute(SERVER_PORT, port)


def start_span(tracer: Tracer, operation: str, model: Any) -> Span:
    name = f"{operation} {model}" if model else operation
    return tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            attrs.GEN_AI_OPERATION_NAME: operation,
            attrs.GEN_AI_PROVIDER_NAME: attrs.PROVIDER_NAME,
        },
    )


@dont_throw
def record_error(span: Span, exc: BaseException) -> None:
    span.set_attribute(ERROR_TYPE, type(exc).__qualname__)
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        span.set_attribute(HTTP_RESPONSE_STATUS_CODE, status_code)
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc) or type(exc).__qualname__))


def input_text_messages(text: str) -> str:
    return json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": text}]}],
        ensure_ascii=False,
    )


def output_text_messages(text: str) -> str:
    return json.dumps(
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": text}],
                "finish_reason": "stop",
            }
        ],
        ensure_ascii=False,
    )
