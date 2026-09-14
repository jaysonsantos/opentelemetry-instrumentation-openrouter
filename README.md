# opentelemetry-instrumentation-openrouter

OpenTelemetry tracing for the official [OpenRouter Python SDK](https://github.com/OpenRouterTeam/python-sdk) (`openrouter` on PyPI).

## What it traces

The instrumentation creates one span of kind `CLIENT` for each call to these SDK methods:

| SDK method | Operation |
| --- | --- |
| `client.tts.create_speech` | `text_to_speech` |
| `client.tts.create_speech_async` | `text_to_speech` |
| `client.stt.create_transcription` | `speech_to_text` |
| `client.stt.create_transcription_async` | `speech_to_text` |
| `client.stt.create_transcription_multipart` | `speech_to_text` |
| `client.stt.create_transcription_multipart_async` | `speech_to_text` |

The span name is `{gen_ai.operation.name} {gen_ai.request.model}`, for example `text_to_speech openai/gpt-4o-mini-tts`.

If the installed SDK version does not have a method, the instrumentation skips that method.

The span is the current span during the HTTP request.
If you also use the httpx instrumentation, the HTTP spans are children of the OpenRouter span.

## Install

```sh
pip install opentelemetry-instrumentation-openrouter
```

The package does not install `openrouter`. To install both, use the `instruments` extra:

```sh
pip install "opentelemetry-instrumentation-openrouter[instruments]"
```

## Usage

### Zero-code instrumentation

The package registers the `openrouter` entry point in the `opentelemetry_instrumentor` group.
`opentelemetry-instrument` loads it automatically:

```sh
opentelemetry-instrument --traces_exporter console python app.py
```

### Manual instrumentation

```python
from openrouter import OpenRouter
from opentelemetry.instrumentation.openrouter import OpenRouterInstrumentor

OpenRouterInstrumentor().instrument()

client = OpenRouter(api_key="...")
response = client.tts.create_speech(
    model="openai/gpt-4o-mini-tts",
    input="Hello",
    voice="alloy",
    response_format="mp3",
)
audio = response.read()  # The span ends here.
```

`instrument()` accepts these keyword arguments:

- `tracer_provider`: the `TracerProvider` to use. The default is the global provider.
- `capture_content`: `True` or `False`. This value overrides the environment variable. The default is `None`.

To remove the instrumentation, call `OpenRouterInstrumentor().uninstrument()`.

## Attributes

### All spans

| Attribute | Example | Notes |
| --- | --- | --- |
| `gen_ai.operation.name` | `text_to_speech` | Custom value. See [Operation names](#operation-names). |
| `gen_ai.provider.name` | `openrouter` | |
| `gen_ai.request.model` | `openai/gpt-4o-mini-tts` | |
| `gen_ai.output.type` | `speech` or `text` | `speech` for TTS, `text` for STT. |
| `server.address` | `openrouter.ai` | From the SDK server URL. |
| `server.port` | `443` | From the SDK server URL. |
| `http.response.status_code` | `200` | TTS: always. STT: only on an HTTP error. |
| `error.type` | `BadRequestResponseError` | Only on failure. The class name of the exception. |

On failure, the span also gets an exception event and the status `ERROR`.
The instrumentation re-raises the original exception unchanged.

### Text-to-speech

| Attribute | Example | Notes |
| --- | --- | --- |
| `openrouter.tts.voice` | `alloy` | |
| `openrouter.tts.response_format` | `mp3` | If the caller does not set it, the value is `pcm` (the SDK default). |
| `openrouter.tts.speed` | `1.2` | Only if the caller sets it. |
| `openrouter.tts.input.characters` | `5` | Length of the input text. |
| `openrouter.tts.output.bytes` | `48213` | Audio bytes that the caller read. |
| `gen_ai.response.time_to_first_chunk` | `0.412` | Seconds from the call start to the first body chunk. |
| `gen_ai.input.messages` | see below | Only when content capture is on. |

### Speech-to-text

| Attribute | Example | Notes |
| --- | --- | --- |
| `openrouter.stt.language` | `en` | Request language hint. |
| `openrouter.stt.response_format` | `verbose_json` | Only if the caller sets it. |
| `openrouter.stt.input.format` | `mp3` | JSON requests only. |
| `openrouter.stt.input.bytes` | `48213` | Size of the input audio. The instrumentation never records the audio. |
| `gen_ai.response.model` | | Only if the response has a `model` field. |
| `openrouter.stt.response.language` | `en` | Language in the response. |
| `openrouter.stt.audio.duration` | `1.9` | Audio duration in seconds, from the response. |
| `gen_ai.usage.input_tokens` | `120` | From `usage.input_tokens`. |
| `gen_ai.usage.output_tokens` | `8` | From `usage.output_tokens`. |
| `openrouter.usage.total_tokens` | `128` | From `usage.total_tokens`. The GenAI conventions have no attribute for it. |
| `openrouter.usage.seconds` | `1.9` | From `usage.seconds`. |
| `openrouter.usage.cost` | `0.00012` | From `usage.cost`, in credits. |
| `gen_ai.output.messages` | see below | Only when content capture is on. |

`openrouter.stt.input.bytes` is available in these cases:

- JSON request: the instrumentation calculates the decoded size of the base64 data.
- Multipart request: the file content is `bytes`, `bytearray`, `memoryview`, `io.BytesIO`, or an object with a `fileno()` method.

## Content capture

Content capture is off by default.
When it is off, the spans do not contain the TTS input text or the STT transcript.

To turn it on, set the environment variable:

```sh
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

The instrumentation reads the variable on each call.
Only the value `true` (not case-sensitive) turns capture on.
The `capture_content` argument of `instrument()` overrides the variable.

When capture is on, the instrumentation sets these attributes as JSON strings:

- TTS, `gen_ai.input.messages`:
  `[{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]`
- STT, `gen_ai.output.messages`:
  `[{"role": "assistant", "parts": [{"type": "text", "content": "Hello"}], "finish_reason": "stop"}]`

## Operation names

The OpenTelemetry GenAI semantic conventions define no operation name for speech.
This package uses two custom values for `gen_ai.operation.name`:

- `text_to_speech`
- `speech_to_text`

If the conventions add names for speech operations, a later release will use them.

## TTS streaming

`create_speech` returns an unread streamed `httpx.Response`.
The span stays open until your code reads or closes the body.

The instrumentation replaces `response.stream` with a proxy.
The proxy counts the bytes and records the time of the first chunk.
It ends the span one time, at the first of these events:

- The body is fully read: `read()`, `aread()`, `iter_bytes()`, `aiter_bytes()`, or other iterators.
- The response is closed: `close()` or `aclose()`.
- An error occurs while the body is read. The span gets the error.

`time_to_first_chunk` includes the time before your code starts to read the body.

> **Warning:** Always read or close the response.
> If you do neither, the span stays open. The span is not exported.
> The SDK also keeps the HTTP connection open until garbage collection.

## Development

```sh
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

`examples/smoke.py` calls the real OpenRouter API. Read the file header before you run it.

## License

Apache-2.0. See `LICENSE`.
