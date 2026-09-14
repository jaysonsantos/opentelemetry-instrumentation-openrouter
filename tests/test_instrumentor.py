from __future__ import annotations

import base64
import json

import httpx
import pytest
from openrouter.stt import STT
from openrouter.tts import TTS
from opentelemetry.instrumentation.utils import suppress_instrumentation

from opentelemetry.instrumentation.openrouter import OpenRouterInstrumentor

from .conftest import CAPTURE_ENV, make_client

METHODS = [
    (TTS, "create_speech"),
    (TTS, "create_speech_async"),
    (STT, "create_transcription"),
    (STT, "create_transcription_async"),
    (STT, "create_transcription_multipart"),
    (STT, "create_transcription_multipart_async"),
]


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/audio/speech"):
        return httpx.Response(
            200, headers={"content-type": "audio/mpeg"}, content=b"audio"
        )
    return httpx.Response(200, json={"text": "Hello there"})


def _run_tts_and_stt(client) -> None:
    client.tts.create_speech(model="m-tts", input="Hello there").read()
    client.stt.create_transcription(
        model="m-stt",
        input_audio={"data": base64.b64encode(b"audio").decode(), "format": "mp3"},
    )


def _messages(exporter) -> tuple:
    tts_span, stt_span = exporter.get_finished_spans()
    return (
        tts_span.attributes.get("gen_ai.input.messages"),
        stt_span.attributes.get("gen_ai.output.messages"),
    )


EXPECTED_INPUT = [
    {"role": "user", "parts": [{"type": "text", "content": "Hello there"}]}
]
EXPECTED_OUTPUT = [
    {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Hello there"}],
        "finish_reason": "stop",
    }
]


def test_dependencies():
    assert OpenRouterInstrumentor().instrumentation_dependencies() == (
        "openrouter >= 1.1.0",
    )


def test_content_capture_off_by_default(instrumented, exporter):
    _run_tts_and_stt(make_client(_handler))
    assert _messages(exporter) == (None, None)


def test_content_capture_on_by_env(instrumented, exporter, monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV, "True")
    _run_tts_and_stt(make_client(_handler))

    input_messages, output_messages = _messages(exporter)
    assert json.loads(input_messages) == EXPECTED_INPUT
    assert json.loads(output_messages) == EXPECTED_OUTPUT


def test_content_capture_kwarg_true_overrides_env(instrument, exporter, monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV, "false")
    instrument(capture_content=True)
    _run_tts_and_stt(make_client(_handler))

    input_messages, output_messages = _messages(exporter)
    assert json.loads(input_messages) == EXPECTED_INPUT
    assert json.loads(output_messages) == EXPECTED_OUTPUT


def test_content_capture_kwarg_false_overrides_env(instrument, exporter, monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV, "true")
    instrument(capture_content=False)
    _run_tts_and_stt(make_client(_handler))
    assert _messages(exporter) == (None, None)


def test_suppressed_instrumentation_creates_no_spans(instrumented, exporter):
    client = make_client(_handler)
    with suppress_instrumentation():
        _run_tts_and_stt(client)
    assert exporter.get_finished_spans() == ()


def test_uninstrument_restores_originals(instrument, exporter):
    originals = {(cls, name): cls.__dict__[name] for cls, name in METHODS}

    instrumentor = instrument()
    for cls, name in METHODS:
        assert cls.__dict__[name] is not originals[(cls, name)]
        assert cls.__dict__[name].__wrapped__ is originals[(cls, name)]

    instrumentor.uninstrument()
    for cls, name in METHODS:
        assert cls.__dict__[name] is originals[(cls, name)]

    _run_tts_and_stt(make_client(_handler))
    assert exporter.get_finished_spans() == ()


def test_missing_method_is_skipped(instrument, exporter, monkeypatch):
    monkeypatch.delattr(STT, "create_transcription_multipart_async")

    instrument()

    assert "create_transcription_multipart_async" not in STT.__dict__
    _run_tts_and_stt(make_client(_handler))
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.parametrize(("cls", "name"), METHODS)
def test_all_methods_exist_in_installed_sdk(cls, name):
    assert callable(getattr(cls, name))
