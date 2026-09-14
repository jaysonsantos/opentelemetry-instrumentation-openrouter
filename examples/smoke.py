"""Live smoke test for the OpenRouter instrumentation. Do not run it in CI.

It calls the real OpenRouter API: text-to-speech, then speech-to-text on the
returned audio. Each run costs a small amount of credit.

Run:

    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
        uv run python examples/smoke.py

Export to OTLP (for example Jaeger) instead of the console:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
        uv run --with opentelemetry-exporter-otlp-proto-http \
        python examples/smoke.py

Environment variables:

* ``OPENROUTER_API_KEY``: if not set, the script reads it from
  ``SMOKE_ENV_FILE`` (default ``~/.config/sparky-home/.env``).
* ``SMOKE_TTS_MODEL``, ``SMOKE_TTS_VOICE``, ``SMOKE_STT_MODEL``,
  ``SMOKE_TEXT``: override the defaults below.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from openrouter import OpenRouter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from opentelemetry.instrumentation.openrouter import OpenRouterInstrumentor

DEFAULT_TTS_MODEL = "microsoft/mai-voice-2-flash"
DEFAULT_TTS_VOICE = "en-US-Harper:MAI-Voice-2"
DEFAULT_STT_MODEL = "microsoft/mai-transcribe-2"
DEFAULT_TEXT = "Hello from Sparky."


def load_api_key() -> None:
    """Put OPENROUTER_API_KEY in the process environment. Never print it."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    env_file = Path(
        os.environ.get("SMOKE_ENV_FILE", "~/.config/sparky-home/.env")
    ).expanduser()
    if env_file.is_file():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[len("export ") :]
            key, sep, value = line.partition("=")
            if sep and key.strip() == "OPENROUTER_API_KEY":
                os.environ["OPENROUTER_API_KEY"] = value.strip().strip("'\"")
                return
    sys.exit(f"OPENROUTER_API_KEY is not set and not found in {env_file}")


def make_tracer_provider() -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "openrouter-smoke"})
    )
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider


def main() -> None:
    load_api_key()
    provider = make_tracer_provider()
    OpenRouterInstrumentor().instrument(tracer_provider=provider)

    tts_model = os.environ.get("SMOKE_TTS_MODEL", DEFAULT_TTS_MODEL)
    tts_voice = os.environ.get("SMOKE_TTS_VOICE", DEFAULT_TTS_VOICE) or None
    stt_model = os.environ.get("SMOKE_STT_MODEL", DEFAULT_STT_MODEL)
    text = os.environ.get("SMOKE_TEXT", DEFAULT_TEXT)

    client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
    try:
        response = client.tts.create_speech(
            model=tts_model,
            input=text,
            voice=tts_voice,
            response_format="mp3",
        )
        audio = response.read()
        print(f"TTS: {len(audio)} bytes of audio", file=sys.stderr)

        result = client.stt.create_transcription(
            model=stt_model,
            input_audio={
                "data": base64.b64encode(audio).decode("ascii"),
                "format": "mp3",
            },
        )
        print(f"STT: {result.text!r}", file=sys.stderr)
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
