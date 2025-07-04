import json
from openai import OpenAI, RateLimitError, BadRequestError
from openai.types.chat import ChatCompletionMessage

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult

from .backend import Backend, BackendResponse


class OpenAIBackend(Backend):
    NAME = 'openai'
    MODELS = {
        "gpt-4o-2024-11-20": {
            "max_context": 128000,
            "cost_per_input_token": 2.5e-06,
            "cost_per_output_token": 10e-06
        },
        "gpt-4o-2024-08-06": {
            "max_context": 128000,
            "cost_per_input_token": 2.5e-06,
            "cost_per_output_token": 10e-06
        },
        "gpt-4o-2024-05-13": {
            "max_context": 128000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 15e-06
        },
        "gpt-4o-mini-2024-07-18": {
            "max_context": 128000,
            "cost_per_input_token": 0.15e-06,
            "cost_per_output_token": 0.6e-06
        },
        "gpt-3.5-turbo-1106": {
            "max_context": 16385,
            "cost_per_input_token": 1e-06,
            "cost_per_output_token": 2e-06
        },
        "gpt-4-1106-preview": {
            "max_context": 128000,
            "cost_per_input_token": 10e-06,
            "cost_per_output_token": 30e-06
        },
        "gpt-4-0125-preview": {
            "max_context": 128000,
            "cost_per_input_token": 10e-06,
            "cost_per_output_token": 30e-06
        },
        "gpt-4-turbo-2024-04-09": {
            "max_context": 128000,
            "cost_per_input_token": 10e-06,
            "cost_per_output_token": 30e-06
        },
        "gpt-4.1-2025-04-14": {
            "max_context": 1000000,
            "cost_per_input_token": 2e-06,
            "cost_per_output_token": 8e-06
        }
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.client = OpenAI(api_key=api_key)
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    @staticmethod
    def get_tool_schema(tool):
        # Based on required OpenAI format, https://platform.openai.com/docs/guides/function-calling
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

    # @backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
    def _call_model(self, messages) -> ChatCompletionMessage:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto", # TODO try "required" here to force a function call
            parallel_tool_calls=False,
            temperature=self.get_param(self.role, "temperature"),
            max_tokens=self.get_param(self.role, "max_tokens")
        )

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
