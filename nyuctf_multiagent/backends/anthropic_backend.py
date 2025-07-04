import json
from anthropic import Anthropic, RateLimitError

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult


from .backend import Backend, BackendResponse

class AnthropicBackend(Backend):
    NAME = "anthropic"
    MODELS = {
        "claude-3-5-sonnet-20241022": {
            "max_context": 200000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06
        },
        "claude-3-5-sonnet-20240620": {
            "max_context": 200000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06
        },
        "claude-3-5-haiku-20241022": {
            "max_context": 200000,
            "cost_per_input_token": 0.8e-06,
            "cost_per_output_token": 4e-06
        },
        "claude-3-7-sonnet-20250219": {
            "max_context": 200000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06
        },
        "claude-sonnet-4-20250514": {
            "max_context": 200000,
            "cost_per_input_token": 3e-06,
            "cost_per_output_token": 15e-06
        }
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.client = Anthropic(api_key=api_key)
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    @staticmethod
    def get_tool_schema(tool):
        # Based on required OpenAI format, https://platform.openai.com/docs/guides/function-calling
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
        return self.client.messages.create(
                model=self.model,
                max_tokens=self.get_param(self.role, "max_tokens"),
                temperature=self.get_param(self.role, "temperature"),
                system=system,
                tools=self.tool_schemas,
                messages=messages)

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
