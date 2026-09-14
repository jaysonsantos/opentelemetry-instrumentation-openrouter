"""Wrappers for ``openrouter.stt.STT`` (speech-to-text)."""

from __future__ import annotations

import io
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.trace import Span, Tracer

from opentelemetry.instrumentation.openrouter import _attributes as attrs
from opentelemetry.instrumentation.openrouter._utils import (
    dont_throw,
    get_field,
    is_content_capture_enabled,
    output_text_messages,
    record_error,
    set_attr,
    set_server_attributes,
    start_span,
)


def _base64_decoded_size(data: str) -> int | None:
    if "," in data[:100] and data.startswith("data:"):
        data = data.split(",", 1)[1]
    length = len(data)
    if length == 0 or length % 4 != 0:
        return None
    return length * 3 // 4 - data[-2:].count("=")


def _content_size(content: Any) -> int | None:
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    if isinstance(content, memoryview):
        return content.nbytes
    if isinstance(content, io.BytesIO):
        return content.getbuffer().nbytes
    fileno = getattr(content, "fileno", None)
    if callable(fileno):
        return os.fstat(fileno()).st_size
    return None


@dont_throw
def _set_json_input_attributes(span: Span, kwargs: dict[str, Any]) -> None:
    input_audio = kwargs.get("input_audio")
    set_attr(
        span,
        attrs.OPENROUTER_STT_INPUT_FORMAT,
        get_field(input_audio, "format_", "format"),
    )
    data = get_field(input_audio, "data")
    if isinstance(data, str):
        set_attr(span, attrs.OPENROUTER_STT_INPUT_BYTES, _base64_decoded_size(data))


@dont_throw
def _set_multipart_input_attributes(span: Span, kwargs: dict[str, Any]) -> None:
    file = kwargs.get("file")
    content = get_field(file, "content")
    set_attr(span, attrs.OPENROUTER_STT_INPUT_BYTES, _content_size(content))


@dont_throw
def _set_request_attributes(
    span: Span, instance: Any, kwargs: dict[str, Any], multipart: bool
) -> None:
    set_attr(span, attrs.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    span.set_attribute(attrs.GEN_AI_OUTPUT_TYPE, attrs.OUTPUT_TYPE_TEXT)
    set_server_attributes(span, instance, kwargs)
    set_attr(span, attrs.OPENROUTER_STT_LANGUAGE, kwargs.get("language"))
    set_attr(span, attrs.OPENROUTER_STT_RESPONSE_FORMAT, kwargs.get("response_format"))
    if multipart:
        _set_multipart_input_attributes(span, kwargs)
    else:
        _set_json_input_attributes(span, kwargs)


@dont_throw
def _set_response_attributes(
    span: Span, result: Any, capture_content: bool | None
) -> None:
    set_attr(span, attrs.GEN_AI_RESPONSE_MODEL, get_field(result, "model"))
    set_attr(
        span, attrs.OPENROUTER_STT_RESPONSE_LANGUAGE, get_field(result, "language")
    )
    set_attr(span, attrs.OPENROUTER_STT_AUDIO_DURATION, get_field(result, "duration"))

    usage = get_field(result, "usage")
    if usage is not None:
        set_attr(
            span, attrs.GEN_AI_USAGE_INPUT_TOKENS, get_field(usage, "input_tokens")
        )
        set_attr(
            span, attrs.GEN_AI_USAGE_OUTPUT_TOKENS, get_field(usage, "output_tokens")
        )
        set_attr(
            span, attrs.OPENROUTER_USAGE_TOTAL_TOKENS, get_field(usage, "total_tokens")
        )
        set_attr(span, attrs.OPENROUTER_USAGE_SECONDS, get_field(usage, "seconds"))
        set_attr(span, attrs.OPENROUTER_USAGE_COST, get_field(usage, "cost"))

    text = get_field(result, "text")
    if isinstance(text, str) and is_content_capture_enabled(capture_content):
        span.set_attribute(attrs.GEN_AI_OUTPUT_MESSAGES, output_text_messages(text))


def create_transcription_wrapper(
    tracer: Tracer, capture_content: bool | None, multipart: bool
):
    def wrapper(wrapped, instance, args, kwargs):
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)

        span = start_span(tracer, attrs.OPERATION_SPEECH_TO_TEXT, kwargs.get("model"))
        _set_request_attributes(span, instance, kwargs, multipart)
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                record_error(span, exc)
                span.end()
                raise
        _set_response_attributes(span, result, capture_content)
        span.end()
        return result

    return wrapper


def create_transcription_async_wrapper(
    tracer: Tracer, capture_content: bool | None, multipart: bool
):
    async def wrapper(wrapped, instance, args, kwargs):
        if not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)

        span = start_span(tracer, attrs.OPERATION_SPEECH_TO_TEXT, kwargs.get("model"))
        _set_request_attributes(span, instance, kwargs, multipart)
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                record_error(span, exc)
                span.end()
                raise
        _set_response_attributes(span, result, capture_content)
        span.end()
        return result

    return wrapper
