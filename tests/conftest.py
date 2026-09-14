from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from openrouter import OpenRouter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from opentelemetry.instrumentation.openrouter import OpenRouterInstrumentor

CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _clear_capture_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CAPTURE_ENV, raising=False)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def instrument(tracer_provider: TracerProvider):
    """Return a function that instruments with optional kwargs."""
    instrumentor = OpenRouterInstrumentor()

    def _instrument(**kwargs) -> OpenRouterInstrumentor:
        instrumentor.instrument(tracer_provider=tracer_provider, **kwargs)
        return instrumentor

    yield _instrument
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.fixture
def instrumented(instrument) -> OpenRouterInstrumentor:
    return instrument()


def make_client(handler: Handler) -> OpenRouter:
    transport = httpx.MockTransport(handler)
    return OpenRouter(
        api_key="test-key",
        client=httpx.Client(transport=transport),
        async_client=httpx.AsyncClient(transport=transport),
    )
