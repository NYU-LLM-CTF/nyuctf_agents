import json
from openai import OpenAI, RateLimitError, BadRequestError
from openai.types.chat import ChatCompletionMessage

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult

from .backend import Backend, BackendResponse


class OpenAIBackend(Backend):
    NAME = 'openai'
    DISABLE_REASONING = True
    MODELS = {
        "gpt-5.6-sol": {
            "max_context": 1050000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 30e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.6-terra": {
            "max_context": 1050000,
            "cost_per_input_token": 2.5e-06,
            "cost_per_output_token": 15e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.6-luna": {
            "max_context": 1050000,
            "cost_per_input_token": 1e-06,
            "cost_per_output_token": 6e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.5": {
            "max_context": 1050000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 30e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.5-pro": {
            "max_context": 1050000,
            "cost_per_input_token": 30e-06,
            "cost_per_output_token": 180e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.4": {
            "max_context": 1050000,
            "cost_per_input_token": 2.5e-06,
            "cost_per_output_token": 15e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.4-pro": {
            "max_context": 1050000,
            "cost_per_input_token": 30e-06,
            "cost_per_output_token": 180e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.4-mini": {
            "max_context": 1050000,
            "cost_per_input_token": 0.75e-06,
            "cost_per_output_token": 4.5e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.4-nano": {
            "max_context": 1050000,
            "cost_per_input_token": 0.20e-06,
            "cost_per_output_token": 1.25e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.3-codex": {
            "max_context": 400000,
            "cost_per_input_token": 1.75e-06,
            "cost_per_output_token": 14e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.2": {
            "max_context": 400000,
            "cost_per_input_token": 1.75e-06,
            "cost_per_output_token": 14e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5.1": {
            "max_context": 400000,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 10e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5": {
            "max_context": 400000,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 10e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5-mini": {
            "max_context": 400000,
            "cost_per_input_token": 0.25e-06,
            "cost_per_output_token": 2e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5-nano": {
            "max_context": 400000,
            "cost_per_input_token": 0.05e-06,
            "cost_per_output_token": 0.40e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "gpt-5-pro": {
            "max_context": 400000,
            "cost_per_input_token": 15e-06,
            "cost_per_output_token": 120e-06,
            "reasoning": True,
            "min_reasoning": "none",
        },
        "o3": {
            "max_context": 200000,
            "cost_per_input_token": 2e-06,
            "cost_per_output_token": 8e-06,
            "reasoning": True,
            "min_reasoning": "low",
        },
        "o3-pro": {
            "max_context": 200000,
            "cost_per_input_token": 20e-06,
            "cost_per_output_token": 80e-06,
            "reasoning": True,
            "min_reasoning": "low",
        },
        "gpt-4.1": {
            "max_context": 1000000,
            "cost_per_input_token": 2e-06,
            "cost_per_output_token": 8e-06,
        },
        "gpt-4.1-mini": {
            "max_context": 1000000,
            "cost_per_input_token": 0.40e-06,
            "cost_per_output_token": 1.60e-06,
        },
        "gpt-4o-mini": {
            "max_context": 128000,
            "cost_per_input_token": 0.15e-06,
            "cost_per_output_token": 0.60e-06,
        },
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.client = OpenAI(api_key=api_key)
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    @staticmethod
    def get_tool_schema(tool):
        return {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": tool.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                    "required": list(tool.REQUIRED_PARAMETERS),
                }
            }
        }

    def _call_model(self, messages) -> ChatCompletionMessage:
        cfg = self.MODELS[self.model]
        is_reasoning = cfg.get("reasoning", False)
        kwargs = dict(
            model=self.model,
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        if is_reasoning:
            kwargs["max_completion_tokens"] = self.get_param(self.role, "max_tokens")
            if self.DISABLE_REASONING and cfg.get("min_reasoning") is not None:
                kwargs["reasoning_effort"] = cfg["min_reasoning"]
        else:
            kwargs["max_tokens"] = self.get_param(self.role, "max_tokens")
            kwargs["temperature"] = self.get_param(self.role, "temperature")
            kwargs["top_p"] = self.get_param(self.role, "top_p")
        return self.client.chat.completions.create(**kwargs)

    def calculate_cost(self, response):
        return self.in_price * response.usage.prompt_tokens + self.out_price * response.usage.completion_tokens

    def send(self, messages):
        formatted_messages = []
        for m in messages:
            if m.role == MessageRole.OBSERVATION:
                msg = {"role": "tool",
                       "content": json.dumps(m.tool_data.result),
                       "tool_call_id": m.tool_data.id}
            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": m.role.value}
                if m.content is not None:
                    msg["content"] = m.content
                if m.tool_data is not None:
                    msg["tool_calls"] = [{"id": m.tool_data.id,
                                          "type": "function",
                                          "function": {
                                              "name": m.tool_data.name,
                                              "arguments": m.tool_data.arguments
                                            }}]
            else:
                msg = {"role": m.role.value, "content": m.content}
            formatted_messages.append(msg)

        try:
            response = self._call_model(formatted_messages)
            cost = self.calculate_cost(response)
            response = response.choices[0].message
        except BadRequestError as e:
            return BackendResponse(error=f"Backend Error: {e}")

        if response.tool_calls and len(response.tool_calls) > 0:
            oai_call = response.tool_calls[0]
            tool_call = ToolCall(name=oai_call.function.name, id=oai_call.id,
                                 arguments=oai_call.function.arguments)
        else:
            tool_call = None

        return BackendResponse(content=response.content, tool_call=tool_call, cost=cost)