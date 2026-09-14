"""Attribute names and values used by this instrumentation.

The GenAI names come from ``opentelemetry-semantic-conventions``. That package
marks the ``gen_ai`` constants as deprecated because the GenAI conventions
moved to a separate repository. That repository publishes no Python package,
so the constants in ``_incubating`` are the only Python source for these names.

The ``openrouter.*`` names are custom. No semantic convention defines them.
"""

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as _gen_ai,
)

# GenAI semantic conventions
GEN_AI_OPERATION_NAME = _gen_ai.GEN_AI_OPERATION_NAME
GEN_AI_PROVIDER_NAME = _gen_ai.GEN_AI_PROVIDER_NAME
GEN_AI_REQUEST_MODEL = _gen_ai.GEN_AI_REQUEST_MODEL
GEN_AI_RESPONSE_ID = _gen_ai.GEN_AI_RESPONSE_ID
GEN_AI_RESPONSE_MODEL = _gen_ai.GEN_AI_RESPONSE_MODEL
GEN_AI_OUTPUT_TYPE = _gen_ai.GEN_AI_OUTPUT_TYPE
GEN_AI_INPUT_MESSAGES = _gen_ai.GEN_AI_INPUT_MESSAGES
GEN_AI_OUTPUT_MESSAGES = _gen_ai.GEN_AI_OUTPUT_MESSAGES
GEN_AI_USAGE_INPUT_TOKENS = _gen_ai.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_USAGE_OUTPUT_TOKENS = _gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS
GEN_AI_TIME_TO_FIRST_CHUNK = _gen_ai.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK

# The GenAI conventions have no provider value for OpenRouter.
PROVIDER_NAME = "openrouter"

# Custom operation names: the GenAI conventions define none for speech.
OPERATION_TEXT_TO_SPEECH = "text_to_speech"
OPERATION_SPEECH_TO_TEXT = "speech_to_text"

OUTPUT_TYPE_SPEECH = _gen_ai.GenAiOutputTypeValues.SPEECH.value
OUTPUT_TYPE_TEXT = _gen_ai.GenAiOutputTypeValues.TEXT.value

# OpenRouter specific attributes
OPENROUTER_TTS_VOICE = "openrouter.tts.voice"
OPENROUTER_TTS_RESPONSE_FORMAT = "openrouter.tts.response_format"
OPENROUTER_TTS_SPEED = "openrouter.tts.speed"
OPENROUTER_TTS_INPUT_CHARACTERS = "openrouter.tts.input.characters"
OPENROUTER_TTS_OUTPUT_BYTES = "openrouter.tts.output.bytes"

OPENROUTER_STT_LANGUAGE = "openrouter.stt.language"
OPENROUTER_STT_RESPONSE_FORMAT = "openrouter.stt.response_format"
OPENROUTER_STT_INPUT_FORMAT = "openrouter.stt.input.format"
OPENROUTER_STT_INPUT_BYTES = "openrouter.stt.input.bytes"
OPENROUTER_STT_RESPONSE_LANGUAGE = "openrouter.stt.response.language"
OPENROUTER_STT_AUDIO_DURATION = "openrouter.stt.audio.duration"

OPENROUTER_USAGE_COST = "openrouter.usage.cost"
OPENROUTER_USAGE_SECONDS = "openrouter.usage.seconds"
OPENROUTER_USAGE_TOTAL_TOKENS = "openrouter.usage.total_tokens"
