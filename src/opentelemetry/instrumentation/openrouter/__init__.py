"""OpenTelemetry instrumentation for the OpenRouter Python SDK.

Usage::

    from opentelemetry.instrumentation.openrouter import OpenRouterInstrumentor

    OpenRouterInstrumentor().instrument()

Supported operations: text-to-speech (``client.tts``) and speech-to-text
(``client.stt``). The method table below is the extension point for chat,
embeddings, and other endpoints.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Tracer, get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.openrouter.package import _instruments
from opentelemetry.instrumentation.openrouter.version import __version__

__all__ = ["OpenRouterInstrumentor", "__version__"]

logger = logging.getLogger(__name__)

# A factory gets (tracer, capture_content) and returns a wrapt wrapper.
WrapperFactory = Callable[[Tracer, "bool | None"], Callable[..., Any]]


@dataclass(frozen=True)
class _Method:
    module: str
    cls: str
    method: str
    factory: WrapperFactory


def _methods() -> list[_Method]:
    # Import here: these modules import httpx, which comes with openrouter.
    from opentelemetry.instrumentation.openrouter import stt, tts

    def transcription(multipart: bool, is_async: bool) -> WrapperFactory:
        make = (
            stt.create_transcription_async_wrapper
            if is_async
            else stt.create_transcription_wrapper
        )
        return lambda tracer, capture: make(tracer, capture, multipart)

    return [
        _Method("openrouter.tts", "TTS", "create_speech", tts.create_speech_wrapper),
        _Method(
            "openrouter.tts",
            "TTS",
            "create_speech_async",
            tts.create_speech_async_wrapper,
        ),
        _Method(
            "openrouter.stt", "STT", "create_transcription", transcription(False, False)
        ),
        _Method(
            "openrouter.stt",
            "STT",
            "create_transcription_async",
            transcription(False, True),
        ),
        _Method(
            "openrouter.stt",
            "STT",
            "create_transcription_multipart",
            transcription(True, False),
        ),
        _Method(
            "openrouter.stt",
            "STT",
            "create_transcription_multipart_async",
            transcription(True, True),
        ),
    ]


class OpenRouterInstrumentor(BaseInstrumentor):
    """Instrumentor for the ``openrouter`` Python SDK.

    Keyword arguments for :meth:`instrument`:

    * ``tracer_provider``: the tracer provider to use.
    * ``capture_content``: ``True`` or ``False`` overrides the
      ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment
      variable. ``None`` (default) reads the variable on each call.
    """

    def __init__(self) -> None:
        super().__init__()
        self._wrapped: list[tuple[type, str]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider=kwargs.get("tracer_provider"),
        )
        capture_content: bool | None = kwargs.get("capture_content")
        self._wrapped = []

        for target in _methods():
            try:
                module = importlib.import_module(target.module)
                cls = getattr(module, target.cls, None)
                if cls is None or not callable(getattr(cls, target.method, None)):
                    logger.debug(
                        "Skip %s.%s.%s: not present in this openrouter version",
                        target.module,
                        target.cls,
                        target.method,
                    )
                    continue
                wrap_function_wrapper(
                    module,
                    f"{target.cls}.{target.method}",
                    target.factory(tracer, capture_content),
                )
                self._wrapped.append((cls, target.method))
            except Exception:
                logger.debug(
                    "Failed to wrap %s.%s.%s",
                    target.module,
                    target.cls,
                    target.method,
                    exc_info=True,
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        for cls, method in self._wrapped:
            unwrap(cls, method)
        self._wrapped = []
