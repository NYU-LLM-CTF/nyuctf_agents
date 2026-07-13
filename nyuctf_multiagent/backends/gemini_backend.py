import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult
import uuid
from .backend import Backend, BackendResponse

class GeminiBackend(Backend):
    NAME = "gemini"
    DISABLE_REASONING = True
    MODELS = {
        "gemini-3.5-flash": {
            "max_context": 1000000,
            "cost_per_input_token": 1.5e-06,
            "cost_per_output_token": 9e-06,
            "min_thinking": "minimal",
        },
        "gemini-3.1-pro-preview": {
            "max_context": 1000000,
            "cost_per_input_token": 2e-06,
            "cost_per_output_token": 12e-06,
            "min_thinking": "low",
        },
        "gemini-3.1-flash-lite": {
            "max_context": 1000000,
            "cost_per_input_token": 0.25e-06,
            "cost_per_output_token": 1.5e-06,
            "min_thinking": "minimal",
        },
        "gemini-3-flash-preview": {
            "max_context": 1000000,
            "cost_per_input_token": 0.5e-06,
            "cost_per_output_token": 3e-06,
            "min_thinking": "minimal",
        },
        "gemini-2.5-pro": {
            "max_context": 1048576,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 10e-06,
            "min_thinking": 128,
        },
        "gemini-2.5-flash": {
            "max_context": 1000000,
            "cost_per_input_token": 0.3e-06,
            "cost_per_output_token": 2.5e-06,
            "min_thinking": 0,
        },
        "gemini-2.5-flash-lite": {
            "max_context": 1000000,
            "cost_per_input_token": 0.1e-06,
            "cost_per_output_token": 0.4e-06,
            "min_thinking": 0,
        },
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tool_schemas = [types.Tool(function_declarations=[
            self.get_tool_schema(tool) for tool in tools.values()
        ])]

    @staticmethod
    def get_tool_schema(tool):
        return types.FunctionDeclaration(
            name=tool.NAME,
            description=tool.DESCRIPTION,
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    n: types.Schema(type=p[0].upper(), description=p[1])
                    for n, p in tool.PARAMETERS.items()
                },
                required=list(tool.REQUIRED_PARAMETERS),
            ),
        )

    def _thinking_config(self):
        if not self.DISABLE_REASONING:
            return None
        val = self.MODELS[self.model].get("min_thinking")
        if val is None:
            return None
        if isinstance(val, str):
            return types.ThinkingConfig(thinking_level=val)
        else:
            return types.ThinkingConfig(thinking_budget=val)

    def _call_model(self, system, messages):
        cfg_kwargs = dict(
            system_instruction=system,
            temperature=self.get_param(self.role, "temperature"),
            max_output_tokens=self.get_param(self.role, "max_tokens"),
            tools=self.tool_schemas,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        thinking = self._thinking_config()
        if thinking is not None:
            cfg_kwargs["thinking_config"] = thinking
        return self.client.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )

    def calculate_cost(self, response):
        return self.in_price * response["usage_metadata"]["prompt_token_count"] + self.out_price * response["usage_metadata"]["candidates_token_count"]

    def send(self, messages):
        formatted_messages = []
        system = None
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system = m.content
                continue
            if m.role == MessageRole.OBSERVATION:
                msg = {"role": "user",
                       "parts": [{"text": str(json.dumps(m.tool_data.result))}]}
            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": "model" if m.role.value == "assistant" else "user",
                       "parts": [{"text": "Assistant has no thought!"}]}
                if m.content is not None and len(m.content) > 0:
                    msg["parts"] = [{"text": m.content}]
                if m.tool_data is not None:
                    msg["parts"] = [{"function_call": {
                                        "name": m.tool_data.name,
                                        "args": m.tool_data.arguments
                                    }}]
            else:
                msg = {"role": "model" if m.role.value == "assistant" else "user",
                       "parts": [{"text": "Assistant has no thought" if m.content is None else str(m.content)}]}
            formatted_messages.append(msg)

        try:
            response = self._call_model(system, formatted_messages).to_json_dict()
            cost = self.calculate_cost(response)
        except genai_errors.APIError as e:
            return BackendResponse(error=f"Backend Error: {e}")

        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return BackendResponse(content=None, tool_call=None, cost=0)

            parts = candidates[0].get("content", {}).get("parts", [])
            content = [m.get('text') for m in parts if isinstance(m, dict) and "text" in m]
            tool_call = [m.get('function_call') for m in parts if isinstance(m, dict) and 'function_call' in m]

        except Exception as e:
            return BackendResponse(content=None, tool_call=None, cost=0)
        if len(content) > 0:
            content = content[0]
        else:
            content = None

        if len(tool_call) > 0:
            tool_call = tool_call[0]
            tool_call = ToolCall(name=tool_call["name"], id=str(uuid.uuid4()),
                                 arguments=tool_call["args"])
        else:
            tool_call = None

        return BackendResponse(content=content, tool_call=tool_call, cost=cost)