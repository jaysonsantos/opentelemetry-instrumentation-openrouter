from __future__ import annotations

import base64
import contextlib
import json
import logging

import httpx
import pytest
from openrouter import errors
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from .conftest import make_client

MODEL = "openai/gpt-4o-mini-tts"
CHUNKS = [b"abc", b"defg", b"hi"]
AUDIO = b"".join(CHUNKS)
GENERATION_ID = "gen-1757846400-abc123"
AUDIO_HEADERS = {"content-type": "audio/mpeg", "x-generation-id": GENERATION_ID}


def _sync_audio_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers=AUDIO_HEADERS, content=iter(CHUNKS))


async def _achunks():
    for chunk in CHUNKS:
        yield chunk


def _async_audio_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers=AUDIO_HEADERS, content=_achunks())


def _speak(client, **kwargs):
    params = {
        "model": MODEL,
        "input": "Hello",
        "voice": "alloy",
        "response_format": "mp3",
    }
    params.update(kwargs)
    return client.tts.create_speech(**params)


async def _speak_async(client, **kwargs):
    params = {
        "model": MODEL,
        "input": "Hello",
        "voice": "alloy",
        "response_format": "mp3",
    }
    params.update(kwargs)
    return await client.tts.create_speech_async(**params)


@pytest.fixture
def no_double_end(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="opentelemetry.sdk.trace")
    yield
    assert "Calling end() on an ended span" not in caplog.text


def _assert_success_span(span, num_bytes: int = len(AUDIO)) -> None:
    assert span.name == f"text_to_speech {MODEL}"
    assert span.kind is SpanKind.CLIENT
    attributes = span.attributes
    assert attributes["gen_ai.operation.name"] == "text_to_speech"
    assert attributes["gen_ai.provider.name"] == "openrouter"
    assert attributes["gen_ai.request.model"] == MODEL
    assert attributes["gen_ai.output.type"] == "speech"
    assert attributes["server.address"] == "openrouter.ai"
    assert attributes["server.port"] == 443
    assert attributes["http.response.status_code"] == 200
    assert attributes["gen_ai.response.id"] == GENERATION_ID
    assert attributes["openrouter.tts.voice"] == "alloy"
    assert attributes["openrouter.tts.response_format"] == "mp3"
    assert attributes["openrouter.tts.input.characters"] == 5
    assert attributes["openrouter.tts.output.bytes"] == num_bytes
    if num_bytes:
        assert attributes["gen_ai.response.time_to_first_chunk"] >= 0
    assert "gen_ai.input.messages" not in attributes
    assert span.status.status_code is StatusCode.UNSET


def test_sync_read_ends_span_after_body(instrumented, exporter, no_double_end):
    client = make_client(_sync_audio_handler)

    response = _speak(client)
    assert exporter.get_finished_spans() == ()

    assert response.read() == AUDIO
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0])

    response.close()
    assert len(exporter.get_finished_spans()) == 1


def test_sync_iter_bytes_ends_span_once(instrumented, exporter, no_double_end):
    client = make_client(_sync_audio_handler)

    response = _speak(client)
    received = b""
    for chunk in response.iter_bytes():
        assert exporter.get_finished_spans() == ()
        received += chunk
    assert received == AUDIO

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0])


def test_sync_close_without_read_ends_span(instrumented, exporter, no_double_end):
    client = make_client(_sync_audio_handler)

    with contextlib.closing(_speak(client)) as response:
        assert response.status_code == 200
        assert exporter.get_finished_spans() == ()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0], num_bytes=0)
    assert "gen_ai.response.time_to_first_chunk" not in spans[0].attributes


def test_sync_partial_read_then_close(instrumented, exporter, no_double_end):
    client = make_client(_sync_audio_handler)

    response = _speak(client)
    for chunk in response.iter_raw():
        assert chunk == CHUNKS[0]
        break
    assert exporter.get_finished_spans() == ()
    response.close()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0], num_bytes=len(CHUNKS[0]))


def test_sync_error_during_body_read(instrumented, exporter, no_double_end):
    def broken_body():
        yield b"abc"
        raise httpx.ReadError("connection lost")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "audio/mpeg"}, content=broken_body()
        )

    client = make_client(handler)
    response = _speak(client)
    with pytest.raises(httpx.ReadError, match="connection lost"):
        response.read()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "ReadError"
    assert span.attributes["openrouter.tts.output.bytes"] == 3
    assert [event.name for event in span.events] == ["exception"]


async def test_async_aread_ends_span_after_body(instrumented, exporter, no_double_end):
    client = make_client(_async_audio_handler)

    response = await _speak_async(client)
    assert exporter.get_finished_spans() == ()

    assert await response.aread() == AUDIO
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0])

    await response.aclose()
    assert len(exporter.get_finished_spans()) == 1


async def test_async_aiter_bytes_ends_span_once(instrumented, exporter, no_double_end):
    client = make_client(_async_audio_handler)

    response = await _speak_async(client)
    received = b""
    async for chunk in response.aiter_bytes():
        assert exporter.get_finished_spans() == ()
        received += chunk
    assert received == AUDIO

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0])


async def test_async_close_without_read_ends_span(
    instrumented, exporter, no_double_end
):
    client = make_client(_async_audio_handler)

    async with contextlib.aclosing(await _speak_async(client)) as response:
        assert response.status_code == 200
        assert exporter.get_finished_spans() == ()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0], num_bytes=0)


def _bad_request_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        400,
        headers={"content-type": "application/json"},
        content=json.dumps({"error": {"code": 400, "message": "bad voice"}}),
    )


def test_sync_http_error(instrumented, exporter):
    client = make_client(_bad_request_handler)

    with pytest.raises(errors.BadRequestResponseError) as exc_info:
        _speak(client)
    assert exc_info.value.status_code == 400

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "BadRequestResponseError"
    assert span.attributes["http.response.status_code"] == 400
    assert "openrouter.tts.output.bytes" not in span.attributes
    assert "gen_ai.response.id" not in span.attributes
    assert [event.name for event in span.events] == ["exception"]


async def test_async_http_error(instrumented, exporter):
    client = make_client(_bad_request_handler)

    with pytest.raises(errors.BadRequestResponseError):
        await _speak_async(client)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "BadRequestResponseError"


def test_error_before_response(instrumented, exporter):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    client = make_client(handler)
    with pytest.raises(httpx.ConnectError):
        _speak(client, retries=None)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "ConnectError"


def test_default_response_format_and_custom_server(instrumented, exporter):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return _sync_audio_handler(request)

    client = make_client(handler)
    response = client.tts.create_speech(
        model=MODEL, input="Hi", server_url="http://localhost:8080/api/v1"
    )
    response.read()

    assert seen["url"] == "http://localhost:8080/api/v1/audio/speech"
    (span,) = exporter.get_finished_spans()
    assert span.attributes["openrouter.tts.response_format"] == "pcm"
    assert span.attributes["server.address"] == "localhost"
    assert span.attributes["server.port"] == 8080
    assert "openrouter.tts.voice" not in span.attributes


def test_span_is_current_during_request_and_nests(
    instrumented, exporter, tracer_provider
):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["span_context"] = trace.get_current_span().get_span_context()
        return _sync_audio_handler(request)

    client = make_client(handler)
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("parent") as parent:
        response = _speak(client)
        assert trace.get_current_span() is parent
        response.read()

    tts_span, parent_span = exporter.get_finished_spans()
    assert parent_span.name == "parent"
    assert tts_span.parent.span_id == parent_span.context.span_id
    assert tts_span.context.trace_id == parent_span.context.trace_id
    assert seen["span_context"].span_id == tts_span.context.span_id


def test_missing_generation_id_header(instrumented, exporter, no_double_end):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "audio/mpeg"}, content=iter(CHUNKS)
        )

    client = make_client(handler)
    _speak(client).read()

    (span,) = exporter.get_finished_spans()
    assert "gen_ai.response.id" not in span.attributes
    assert span.attributes["openrouter.tts.output.bytes"] == len(AUDIO)


def test_capture_content_records_input_and_references_without_audio(
    instrument, exporter, no_double_end
):
    instrument(capture_content=True)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _sync_audio_handler(request)

    reference_audio = base64.b64encode(b"reference audio bytes").decode("ascii")
    client = make_client(handler)
    _speak(
        client,
        input_references=[
            {"type": "text", "text": "Reference transcript"},
            {
                "type": "input_audio",
                "input_audio": {"data": reference_audio, "format": "mp3"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": f"data:audio/wav;base64,{reference_audio}"},
            },
        ],
    ).read()

    assert len(seen["body"]["input_references"]) == 3
    (span,) = exporter.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "Hello"},
                {"type": "text", "content": "Reference transcript"},
                {"type": "blob", "modality": "audio", "mime_type": "audio/mpeg"},
                {"type": "blob", "modality": "audio", "mime_type": "audio/wav"},
            ],
        }
    ]
    assert "gen_ai.output.messages" not in span.attributes
    for value in span.attributes.values():
        assert reference_audio not in str(value)
        assert AUDIO.decode("ascii") not in str(value)


def test_preloaded_body_ends_span_at_once(instrumented, exporter, no_double_end):
    def handler(request: httpx.Request) -> httpx.Response:
        # bytes content: httpx reads the body in the constructor.
        return httpx.Response(200, headers=AUDIO_HEADERS, content=AUDIO)

    client = make_client(handler)
    response = _speak(client)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_success_span(spans[0])

    assert response.read() == AUDIO
    response.close()
    assert len(exporter.get_finished_spans()) == 1
