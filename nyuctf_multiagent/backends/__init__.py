from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .openrouter_backend import OpenRouterBackend
from .together_backend import TogetherBackend
from .gemini_backend import GeminiBackend
from .backend import Role

BACKENDS = [OpenAIBackend, OpenRouterBackend, AnthropicBackend, TogetherBackend, GeminiBackend]
MODELS = {m: b for b in BACKENDS for m in b.MODELS}
