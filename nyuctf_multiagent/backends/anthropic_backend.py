import json
from anthropic import Anthropic, RateLimitError

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult


from .backend import Backend, BackendResponse

class AnthropicBackend(Backend):
    NAME = "anthropic"
    DISABLE_REASONING = True
    NO_SAMPLING_MODELS = {
        "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5",
        "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
    }
    MODELS = {
        "claude-opus-4-8": {
            "max_context": 1000000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 25e-06,
            "min_effort": "low",
        },
        "claude-sonnet-5": {
            "max_context": 1000000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06,
            "min_effort": "low",
        },
        "claude-fable-5": {
            "max_context": 1000000,
            "cost_per_input_token": 10e-06,
            "cost_per_output_token": 50e-06,
            "min_effort": "low",
        },
        "claude-haiku-4-5-20251001": {
            "max_context": 200000,
            "cost_per_input_token": 1e-06,
            "cost_per_output_token": 5e-06,
        },
        "claude-opus-4-7": {
            "max_context": 1000000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 25e-06,
            "min_effort": "low",
        },
        "claude-opus-4-6": {
            "max_context": 1000000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 25e-06,
            "min_effort": "low",
        },
        "claude-sonnet-4-6": {
            "max_context": 1000000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06,
            "min_effort": "low",
        },
        "claude-sonnet-4-5-20250929": {
            "max_context": 200000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06,
        },
        "claude-opus-4-5-20251101": {
            "max_context": 200000,
            "cost_per_input_token": 5e-06,
            "cost_per_output_token": 25e-06,
        },
        "claude-opus-4-1-20250805": {
            "max_context": 200000,
            "cost_per_input_token": 15e-06,
            "cost_per_output_token": 75e-06,
        },
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.client = Anthropic(api_key=api_key)
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    @staticmethod
    def get_tool_schema(tool):
        return {
            "name": tool.NAME,
            "description": tool.DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                "required": list(tool.REQUIRED_PARAMETERS),
            }
        }

    def calculate_cost(self, response):
        return self.in_price * response.usage.input_tokens + self.out_price * response.usage.output_tokens

    def _call_model(self, system, messages):
        kwargs = dict(
            model=self.model,
            max_tokens=self.get_param(self.role, "max_tokens"),
            system=system,
            tools=self.tool_schemas,
            messages=messages,
        )
        if self.model not in self.NO_SAMPLING_MODELS:
            kwargs["temperature"] = self.get_param(self.role, "temperature")
        if self.DISABLE_REASONING:
            effort = self.MODELS[self.model].get("min_effort")
            if effort is not None:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": effort}
        return self.client.messages.create(**kwargs)

    def send(self, messages):
        formatted_messages = []
        system = None
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system = m.content
                continue
            if m.role == MessageRole.OBSERVATION:
                msg = {"role": "user",
                       "content": [{
                           "type": "tool_result",
                           "tool_use_id": m.tool_data.id,
                           "content": json.dumps(m.tool_data.result)
                        }]}
            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": m.role.value, "content": []}
                if m.content is not None:
                    msg["content"].append({"type": "text", "text": m.content})
                if m.tool_data is not None:
                    msg["content"].append({"type": "tool_use",
                                           "id": m.tool_data.id,
                                           "name": m.tool_data.name,
                                           "input": m.tool_data.arguments})
            else:
                msg = {"role": m.role.value, "content": [{"type": "text", "text": m.content}]}
            formatted_messages.append(msg)

        try:
            response = self._call_model(system, formatted_messages)
            cost = self.calculate_cost(response)
        except RateLimitError as e:
            return BackendResponse(error=f"Backend Error: {e}")

        content = [m for m in response.content if m.type == "text"]
        tool_call = [m for m in response.content if m.type == "tool_use"]
        if len(content) > 0:
            content = content[0].text
        else:
            content = None

        if len(tool_call) > 0:
            tool_call = tool_call[0]
            tool_call = ToolCall(name=tool_call.name, id=tool_call.id,
                                 arguments=tool_call.input)
        else:
            tool_call = None

        return BackendResponse(content=content, tool_call=tool_call, cost=cost)