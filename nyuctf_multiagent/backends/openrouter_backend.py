from openai import OpenAI

from .backend import Backend
from .openai_backend import OpenAIBackend

class OpenRouterBackend(OpenAIBackend):
    NAME = 'openrouter'
    PREFIX = "openrouter/"
    DISABLE_REASONING = True
    MODELS = {
        "openrouter/anthropic/claude-opus-4.8": {
            "max_context": 1000000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 25e-06,
            "disable_thinking": True,
        },
        "openrouter/openai/gpt-5.6-sol": {
            "max_context": 1050000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 30e-06,
            "disable_thinking": True,
        },
        "openrouter/z-ai/glm-5.2": {
            "max_context": 1048576,
            "cost_per_input_token": 0.42e-06,
            "cost_per_output_token": 1.32e-06,
            "disable_thinking": True,
        },
        "openrouter/deepseek/deepseek-v4-pro": {
            "max_context": 1048576,
            "cost_per_input_token": 0.435e-06,
            "cost_per_output_token": 0.87e-06,
            "disable_thinking": True,
        },
        "openrouter/qwen/qwen3.7-max": {
            "max_context": 1000000,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 3.75e-06,
            "disable_thinking": True,
        },
        "openrouter/google/gemini-3.1-pro-preview": {
            "max_context": 1048576,
            "cost_per_input_token": 2e-06,
            "cost_per_output_token": 12e-06,
            "disable_thinking": False,
        },
    }

    def __init__(self, role, model, tools, api_key, config):
        Backend.__init__(self, role, model, tools, config)
        self.api_model = model[len(self.PREFIX):] if model.startswith(self.PREFIX) else model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    def _call_model(self, messages):
        kwargs = dict(
            model=self.api_model,
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto",
            max_tokens=self.get_param(self.role, "max_tokens"),
        )
        temp = self.get_param(self.role, "temperature")
        top_p = self.get_param(self.role, "top_p")
        if temp is not None:
            kwargs["temperature"] = temp
        if top_p is not None:
            kwargs["top_p"] = top_p
        effort = "none" if self.MODELS[self.model].get("disable_thinking") else "minimal"
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}

        return self.client.chat.completions.create(**kwargs)