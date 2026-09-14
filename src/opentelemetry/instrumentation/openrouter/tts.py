"""Wrappers for ``openrouter.tts.TTS`` (text-to-speech).

``create_speech`` returns an unread, streamed ``httpx.Response``. The span
stays open until the caller consumes or closes the body. To do that, the
wrapper replaces ``response.stream`` with a proxy. Every httpx read path
(``read``, ``iter_bytes``, ``iter_raw``, ``close`` and the async variants)
goes through ``response.stream``, so the proxy sees all of them.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from opentelemetry import trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.semconv.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.trace import Span, Tracer

from opentelemetry.instrumentation.openrouter import _attributes as attrs
from opentelemetry.instrumentation.openrouter._utils import (
    dont_throw,
    input_text_messages,
    is_content_capture_enabled,
    record_error,
    set_attr,
    set_server_attributes,
    start_span,
)

# The SDK sends "pcm" when the caller does not pass ``response_format``.
_DEFAULT_RESPONSE_FORMAT = "pcm"


class _StreamSpan:
    """Holds the span state for one streamed body. Ends the span once."""

    __slots__ = ("_ended", "_first_chunk_at", "_num_bytes", "_span", "_started_at")

    def __init__(self, span: Span, started_at: float) -> None:
        self._span = span
        self._started_at = started_at
        self._num_bytes = 0
        self._first_chunk_at: float | None = None
        self._ended = False

    def on_chunk(self, chunk: bytes) -> None:
        if self._first_chunk_at is None:
            self._first_chunk_at = time.monotonic()
        self._num_bytes += len(chunk)

    def finish(self, exc: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        self._set_final_attributes(exc)
        self._span.end()

    @dont_throw
    def _set_final_attributes(self, exc: BaseException | None) -> None:
        self._span.set_attribute(attrs.OPENROUTER_TTS_OUTPUT_BYTES, self._num_bytes)
        if self._first_chunk_at is not None:
            self._span.set_attribute(
                attrs.GEN_AI_TIME_TO_FIRST_CHUNK,
                self._first_chunk_at - self._started_at,
            )
        if exc is not None:
            record_error(self._span, exc)


class _TracedSyncByteStream(httpx.SyncByteStream):
    def __init__(self, wrapped: httpx.SyncByteStream, state: _StreamSpan) -> None:
        self._wrapped = wrapped
        self._state = state

    def __iter__(self):
        try:
            for chunk in self._wrapped:
                self._state.on_chunk(chunk)
                yield chunk
        except Exception as exc:
            self._state.finish(exc)
            raise
        self._state.finish()

    def close(self) -> None:
        try:
            self._wrapped.close()
        except Exception as exc:
            self._state.finish(exc)
            raise
        self._state.finish()


class _TracedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, wrapped: httpx.AsyncByteStream, state: _StreamSpan) -> None:
        self._wrapped = wrapped
        self._state = state

    async def __aiter__(self):
        try:
            async for chunk in self._wrapped:
                self._state.on_chunk(chunk)
                yield chunk
        except Exception as exc:
            self._state.finish(exc)
            raise
        self._state.finish()

    async def aclose(self) -> None:
        try:
            await self._wrapped.aclose()
        except Exception as exc:
            self._state.finish(exc)
            raise
        self._state.finish()


@dont_throw
def _set_request_attributes(
    span: Span, instance: Any, kwargs: dict[str, Any], capture_content: bool | None
) -> None:
    set_attr(span, attrs.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    span.set_attribute(attrs.GEN_AI_OUTPUT_TYPE, attrs.OUTPUT_TYPE_SPEECH)
    set_server_attributes(span, instance, kwargs)
    set_attr(span, attrs.OPENROUTER_TTS_VOICE, kwargs.get("voice"))
    set_attr(
        span,
        attrs.OPENROUTER_TTS_RESPONSE_FORMAT,
        kwargs.get("response_format", _DEFAULT_RESPONSE_FORMAT),
    )
    set_attr(span, attrs.OPENROUTER_TTS_SPEED, kwargs.get("speed"))
    text = kwargs.get("input")
    if isinstance(text, str):
        span.set_attribute(attrs.OPENROUTER_TTS_INPUT_CHARACTERS, len(text))
        if is_content_capture_enabled(capture_content):
            span.set_attribute(attrs.GEN_AI_INPUT_MESSAGES, input_text_messages(text))


def _attach_stream(
    response: Any, span: Span, started_at: float, is_async: bool
) -> None:
    """Hand the span to the response body, or end it now if that is not possible.

    The span ends now when the body is already in memory (for example a
    transport that builds ``httpx.Response(content=...)``), when the response
    is already closed, or when the stream type is unexpected.
    """
    state = _StreamSpan(span, started_at)
    try:
        set_attr(span, HTTP_RESPONSE_STATUS_CODE, response.status_code)
        content = getattr(response, "_content", None)
        if isinstance(content, bytes):
            # read() and iter_bytes() return this buffer; the stream is unused.
            state.on_chunk(content)
        elif not (response.is_closed or response.is_stream_consumed):
            stream = response.stream
            if is_async and isinstance(stream, httpx.AsyncByteStream):
                response.stream = _TracedAsyncByteStream(stream, state)
                return
            if not is_async and isinstance(stream, httpx.SyncByteStream):
                response.stream = _TracedSyncByteStream(stream, state)
                return
    except Exception:
        pass
    state.finish()


def create_speech_wrapper(tracer: Tracer, capture_content: bool | None):
    def wrapper(wrapped, instance, args, kwargs):
        if not is_instrumentation_enabled():
            return wrapped(*args, **kwargs)

        span = start_span(tracer, attrs.OPERATION_TEXT_TO_SPEECH, kwargs.get("model"))
        _set_request_attributes(span, instance, kwargs, capture_content)
        started_at = time.monotonic()
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exc:
                record_error(span, exc)
                span.end()
                raise
        _attach_stream(response, span, started_at, is_async=False)
        return response

    return wrapper


def create_speech_async_wrapper(tracer: Tracer, capture_content: bool | None):
    async def wrapper(wrapped, instance, args, kwargs):
        if not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)

        span = start_span(tracer, attrs.OPERATION_TEXT_TO_SPEECH, kwargs.get("model"))
        _set_request_attributes(span, instance, kwargs, capture_content)
        started_at = time.monotonic()
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exc:
                record_error(span, exc)
                span.end()
                raise
        _attach_stream(response, span, started_at, is_async=True)
        return response

    return wrapper
