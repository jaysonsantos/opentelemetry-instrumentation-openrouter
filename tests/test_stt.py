from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from openrouter import errors
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from opentelemetry.instrumentation.openrouter import stt as stt_module

from .conftest import make_client

MODEL = "openai/whisper-1"
AUDIO = b"\x00\x01" * 21  # 42 bytes
TRANSCRIPT = "hello world"
STT_BODY = {
    "text": TRANSCRIPT,
    "language": "en",
    "duration": 1.25,
    "usage": {
        "cost": 0.0012,
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
        "seconds": 1.25,
    },
}


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=STT_BODY)


def _json_kwargs() -> dict:
    return {
        "model": MODEL,
        "language": "en",
        "input_audio": {
            "data": base64.b64encode(AUDIO).decode("ascii"),
            "format": "wav",
        },
    }


def _multipart_kwargs(content=AUDIO) -> dict:
    return {
        "model": MODEL,
        "language": "en",
        "file": {
            "file_name": "audio.wav",
            "content": content,
            "content_type": "audio/wav",
        },
    }


def _assert_success_span(span) -> None:
    assert span.name == f"speech_to_text {MODEL}"
    assert span.kind is SpanKind.CLIENT
    attributes = span.attributes
    assert attributes["gen_ai.operation.name"] == "speech_to_text"
    assert attributes["gen_ai.provider.name"] == "openrouter"
    assert attributes["gen_ai.request.model"] == MODEL
    assert attributes["gen_ai.output.type"] == "text"
    assert attributes["server.address"] == "openrouter.ai"
    assert attributes["server.port"] == 443
    assert attributes["openrouter.stt.language"] == "en"
    assert attributes["openrouter.stt.input.bytes"] == len(AUDIO)
    assert attributes["openrouter.stt.response.language"] == "en"
    assert attributes["openrouter.stt.audio.duration"] == 1.25
    assert attributes["gen_ai.usage.input_tokens"] == 10
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert attributes["openrouter.usage.total_tokens"] == 13
    assert attributes["openrouter.usage.seconds"] == 1.25
    assert attributes["openrouter.usage.cost"] == 0.0012
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.output.messages" not in attributes
    assert "gen_ai.response.model" not in attributes
    assert span.status.status_code is StatusCode.UNSET
    for value in attributes.values():
        assert base64.b64encode(AUDIO).decode("ascii") not in str(value)


def test_json_sync(instrumented, exporter):
    client = make_client(_ok_handler)

    result = client.stt.create_transcription(**_json_kwargs())

    assert result.text == TRANSCRIPT
    (span,) = exporter.get_finished_spans()
    _assert_success_span(span)
    assert span.attributes["openrouter.stt.input.format"] == "wav"


async def test_json_async(instrumented, exporter):
    client = make_client(_ok_handler)

    result = await client.stt.create_transcription_async(**_json_kwargs())

    assert result.text == TRANSCRIPT
    (span,) = exporter.get_finished_spans()
    _assert_success_span(span)


def test_multipart_sync(instrumented, exporter):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content_type"] = request.headers["content-type"]
        return _ok_handler(request)

    client = make_client(handler)

    result = client.stt.create_transcription_multipart(**_multipart_kwargs())

    assert result.text == TRANSCRIPT
    assert seen["content_type"].startswith("multipart/form-data")
    (span,) = exporter.get_finished_spans()
    _assert_success_span(span)


async def test_multipart_async(instrumented, exporter):
    client = make_client(_ok_handler)

    result = await client.stt.create_transcription_multipart_async(
        **_multipart_kwargs(content=io.BytesIO(AUDIO))
    )

    assert result.text == TRANSCRIPT
    (span,) = exporter.get_finished_spans()
    _assert_success_span(span)


EXPECTED_OUTPUT_MESSAGES = [
    {
        "role": "assistant",
        "parts": [{"type": "text", "content": TRANSCRIPT}],
        "finish_reason": "stop",
    }
]


def _expected_input_messages(mime_type: str) -> list:
    return [
        {
            "role": "user",
            "parts": [{"type": "blob", "modality": "audio", "mime_type": mime_type}],
        }
    ]


def _assert_no_audio(span) -> None:
    encoded = base64.b64encode(AUDIO).decode("ascii")
    for value in span.attributes.values():
        assert encoded not in str(value)


def test_capture_content_json_records_audio_without_bytes(instrument, exporter):
    instrument(capture_content=True)
    client = make_client(_ok_handler)

    client.stt.create_transcription(**_json_kwargs())

    (span,) = exporter.get_finished_spans()
    assert json.loads(
        span.attributes["gen_ai.input.messages"]
    ) == _expected_input_messages("audio/wav")
    assert json.loads(span.attributes["gen_ai.output.messages"]) == (
        EXPECTED_OUTPUT_MESSAGES
    )
    _assert_no_audio(span)


async def test_capture_content_multipart_records_audio_without_bytes(
    instrument, exporter
):
    instrument(capture_content=True)
    client = make_client(_ok_handler)
    kwargs = _multipart_kwargs()
    kwargs["file"]["content_type"] = "audio/ogg"

    await client.stt.create_transcription_multipart_async(**kwargs)

    (span,) = exporter.get_finished_spans()
    assert json.loads(
        span.attributes["gen_ai.input.messages"]
    ) == _expected_input_messages("audio/ogg")
    assert json.loads(span.attributes["gen_ai.output.messages"]) == (
        EXPECTED_OUTPUT_MESSAGES
    )
    _assert_no_audio(span)


def test_capture_content_records_input_on_error(instrument, exporter):
    instrument(capture_content=True)
    client = make_client(_unauthorized_handler)

    with pytest.raises(errors.UnauthorizedResponseError):
        client.stt.create_transcription(**_json_kwargs())

    (span,) = exporter.get_finished_spans()
    assert json.loads(
        span.attributes["gen_ai.input.messages"]
    ) == _expected_input_messages("audio/wav")
    assert "gen_ai.output.messages" not in span.attributes


def _unauthorized_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        401,
        headers={"content-type": "application/json"},
        content=json.dumps({"error": {"code": 401, "message": "No auth"}}),
    )


def test_error_sync(instrumented, exporter):
    client = make_client(_unauthorized_handler)

    with pytest.raises(errors.UnauthorizedResponseError) as exc_info:
        client.stt.create_transcription(**_json_kwargs())
    assert exc_info.value.status_code == 401

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "UnauthorizedResponseError"
    assert span.attributes["http.response.status_code"] == 401
    assert "openrouter.usage.cost" not in span.attributes
    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"].endswith("UnauthorizedResponseError")


async def test_error_async(instrumented, exporter):
    client = make_client(_unauthorized_handler)

    with pytest.raises(errors.UnauthorizedResponseError):
        await client.stt.create_transcription_multipart_async(**_multipart_kwargs())

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR


def test_span_is_current_during_request_and_nests(
    instrumented, exporter, tracer_provider
):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["span_context"] = trace.get_current_span().get_span_context()
        return _ok_handler(request)

    client = make_client(handler)
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("parent") as parent:
        client.stt.create_transcription(**_json_kwargs())
        assert trace.get_current_span() is parent

    stt_span, parent_span = exporter.get_finished_spans()
    assert stt_span.parent.span_id == parent_span.context.span_id
    assert seen["span_context"].span_id == stt_span.context.span_id


def test_instrumentation_error_does_not_break_call(instrumented, exporter, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("instrumentation bug")

    monkeypatch.setattr(stt_module, "get_field", boom)
    client = make_client(_ok_handler)

    result = client.stt.create_transcription(**_json_kwargs())

    assert result.text == TRANSCRIPT
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET
