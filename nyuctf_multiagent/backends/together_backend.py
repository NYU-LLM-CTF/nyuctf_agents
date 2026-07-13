import json
from together import Together
from together.error import InvalidRequestError, RateLimitError
from together.types.chat_completions import ChatCompletionResponse

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult

from .backend import Backend, BackendResponse

class TogetherBackend(Backend):
    NAME = "together"
    PREFIX = "together/"
    DISABLE_REASONING = True

    MODELS = {
        "together/deepseek-ai/DeepSeek-V4-Pro": {
            "max_context": 524288,
            "cost_per_input_token": 1.74e-06,
            "cost_per_output_token": 3.48e-06,
            "disable_thinking": True,
        },
        "together/zai-org/GLM-5.2": {
            "max_context": 131072,
            "cost_per_input_token": 1.40e-06,
            "cost_per_output_token": 4.40e-06,
            "disable_thinking": True,
        },
        "together/zai-org/GLM-5.1": {
            "max_context": 131072,
            "cost_per_input_token": 1.40e-06,
            "cost_per_output_token": 4.40e-06,
            "disable_thinking": True,
        },
        "together/moonshotai/Kimi-K2.7-Code": {
            "max_context": 262144,
            "cost_per_input_token": 0.95e-06,
            "cost_per_output_token": 4.00e-06,
            "disable_thinking": True,
        },
        "together/moonshotai/Kimi-K2.6": {
            "max_context": 262144,
            "cost_per_input_token": 1.20e-06,
            "cost_per_output_token": 4.50e-06,
            "disable_thinking": True,
        },
        "together/Qwen/Qwen3.7-Max": {
            "max_context": 262144,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 3.75e-06,
            "disable_thinking": True,
        },
        "together/Qwen/Qwen3.7-Plus": {
            "max_context": 262144,
            "cost_per_input_token": 0.32e-06,
            "cost_per_output_token": 1.28e-06,
            "disable_thinking": True,
        },
        "together/Qwen/Qwen3.6-Plus": {
            "max_context": 262144,
            "cost_per_input_token": 0.50e-06,
            "cost_per_output_token": 3.00e-06,
            "disable_thinking": True,
        },
        "together/Qwen/Qwen3.5-397B-A17B": {
            "max_context": 262144,
            "cost_per_input_token": 0.60e-06,
            "cost_per_output_token": 3.60e-06,
            "disable_thinking": True,
        },
        "together/Qwen/Qwen3.5-9B": {
            "max_context": 262144,
            "cost_per_input_token": 0.17e-06,
            "cost_per_output_token": 0.25e-06,
            "disable_thinking": True,
        },
        "together/MiniMaxAI/MiniMax-M3": {
            "max_context": 1000000,
            "cost_per_input_token": 0.30e-06,
            "cost_per_output_token": 1.20e-06,
            "disable_thinking": True,
        },
        "together/MiniMaxAI/MiniMax-M2.7": {
            "max_context": 1000000,
            "cost_per_input_token": 0.30e-06,
            "cost_per_output_token": 1.20e-06,
            "disable_thinking": True,
        },
        "together/deepcogito/cogito-v2.1-671B": {
            "max_context": 131072,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 1.25e-06,
            "disable_thinking": True,
        },
        "together/google/gemma-4-31b-it": {
            "max_context": 131072,
            "cost_per_input_token": 0.39e-06,
            "cost_per_output_token": 0.97e-06,
        },
        "together/nvidia/Nemotron-3-Ultra": {
            "max_context": 1000000,
            "cost_per_input_token": 0.60e-06,
            "cost_per_output_token": 3.60e-06,
            "disable_thinking": True,
        },
        "together/openai/gpt-oss-120b": {
            "max_context": 131072,
            "cost_per_input_token": 0.15e-06,
            "cost_per_output_token": 0.60e-06,
        },
        "together/openai/gpt-oss-20b": {
            "max_context": 131072,
            "cost_per_input_token": 0.05e-06,
            "cost_per_output_token": 0.20e-06,
        },
        "together/meta-llama/Llama-3.3-70B-Instruct-Turbo": {
            "max_context": 131072,
            "cost_per_input_token": 1.04e-06,
            "cost_per_output_token": 1.04e-06,
        },
        "together/meta-llama/Llama-3-8b-instruct-lite": {
            "max_context": 8192,
            "cost_per_input_token": 0.14e-06,
            "cost_per_output_token": 0.14e-06,
        },
        "together/Qwen/Qwen2.5-7B-Instruct-Turbo": {
            "max_context": 32768,
            "cost_per_input_token": 0.30e-06,
            "cost_per_output_token": 0.30e-06,
        },
        "together/Qwen/Qwen3-235B-A22B-Instruct-2507-fp8": {
            "max_context": 262144,
            "cost_per_input_token": 0.20e-06,
            "cost_per_output_token": 0.60e-06,
        },
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.api_model = model[len(self.PREFIX):] if model.startswith(self.PREFIX) else model
        self.client = Together(api_key=api_key)
        if self.get_param(self.role, "strict"):
            self.tool_schemas = [self.get_tool_schema_strict(tool) for tool in tools.values()]
        else:
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
                },
            }
        }

    @staticmethod
    def get_tool_schema_strict(tool):
        schema = {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": tool.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                    "required": list(tool.PARAMETERS.keys()),
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        for propname, prop in schema["function"]["parameters"]["properties"].items():
            if propname not in tool.REQUIRED_PARAMETERS:
                prop["type"] = [prop["type"], "null"]
        return schema

    def _call_model(self, messages) -> ChatCompletionResponse:
        kwargs = dict(
            model=self.api_model,
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto",
            temperature=self.get_param(self.role, "temperature"),
            max_tokens=self.get_param(self.role, "max_tokens"),
        )
        if self.DISABLE_REASONING and self.MODELS[self.model].get("disable_thinking"):
            kwargs["chat_template_kwargs"] = {"thinking": False}
        return self.client.chat.completions.create(**kwargs)

    def calculate_cost(self, response):
        return self.in_price * response.usage.prompt_tokens + self.out_price * response.usage.completion_tokens

    def send(self, messages):
        formatted_messages = []
        for m in messages:
            if m.role == MessageRole.OBSERVATION:
                msg = {"role": "tool",
                       "content": json.dumps(m.tool_data.result),
                       "tool_call_id": m.tool_data.id,
                       "name": m.tool_data.name}
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
        except InvalidRequestError as e:
            return BackendResponse(error=f"Backend Error: {e}")

        if response.tool_calls and len(response.tool_calls) > 0:
            f_call = response.tool_calls[0]
            tool_call = ToolCall(name=f_call.function.name, id=f_call.id,
                                 arguments=f_call.function.arguments)
        else:
            tool_call = None

        return BackendResponse(content=response.content, tool_call=tool_call, cost=cost)