"""Attribute names and values used by this instrumentation.

The GenAI names are plain strings on purpose. In recent
``opentelemetry-semantic-conventions`` releases the ``gen_ai`` constants are
marked deprecated (they moved to a separate GenAI conventions repository).
Plain strings keep the emitted names stable across semconv releases.
"""

# GenAI semantic conventions
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

PROVIDER_NAME = "openrouter"

# Custom operation names: the GenAI conventions define none for speech.
OPERATION_TEXT_TO_SPEECH = "text_to_speech"
OPERATION_SPEECH_TO_TEXT = "speech_to_text"

OUTPUT_TYPE_SPEECH = "speech"
OUTPUT_TYPE_TEXT = "text"

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
