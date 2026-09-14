# AGENTS.md

This file gives instructions to coding agents that work in this repository.
`CLAUDE.md` is a symlink to this file.

## Project

`opentelemetry-instrumentation-openrouter` traces the official OpenRouter Python SDK (`openrouter` on PyPI).
The package supports text-to-speech (`client.tts`) and speech-to-text (`client.stt`).

## Layout

- `src/opentelemetry/instrumentation/openrouter/__init__.py`: `OpenRouterInstrumentor` and the method table.
- `src/opentelemetry/instrumentation/openrouter/tts.py`: TTS wrappers and the stream proxies.
- `src/opentelemetry/instrumentation/openrouter/stt.py`: STT wrappers.
- `src/opentelemetry/instrumentation/openrouter/_attributes.py`: attribute names.
- `src/opentelemetry/instrumentation/openrouter/_utils.py`: shared helpers (`dont_throw`, span start, errors).
- `tests/`: unit tests. The tests use `httpx.MockTransport`. They do not use the network.
- `examples/smoke.py`: live test against the real API. Do not run it in CI.

`src/opentelemetry` and `src/opentelemetry/instrumentation` are namespace packages.
Do not add `__init__.py` files to them.

## Commands

Use `uv` for all tasks.

- Install: `uv sync`
- Lint: `uv run ruff check`
- Format check: `uv run ruff format --check`
- Format: `uv run ruff format`
- Test: `uv run pytest`

Run lint, format check, and tests before each commit.

## Rules

- Instrumentation errors must not break the SDK call. Put attribute code in a function with the `dont_throw` decorator.
- Re-raise SDK exceptions unchanged.
- Never record audio bytes or secrets on a span.
- Use attribute names from `opentelemetry-semantic-conventions` when a constant exists. Add them to `_attributes.py`. Use a custom `openrouter.*` name only when no constant exists.
- Record message content only when content capture is on.
- Keep the TTS span open until the body is read or closed. Do not add `__del__` methods.
- If a method is not present in the installed SDK version, skip it. Do not fail.
- To add an endpoint (for example chat or embeddings), add a module with wrapper factories. Then add rows to `_methods()` in `__init__.py`.
- Do not import `httpx` or `openrouter` at module level in `__init__.py`. The entry point must load without them.

## Written artifacts

Write the README, this file, and commit message bodies in ASD-STE100 Simplified Technical English.
