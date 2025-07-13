import json
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult
import uuid
from .backend import Backend, BackendResponse

class GeminiBackend(Backend):
    NAME = "gemini"
    MODELS = {
        "gemini-2.5-pro": {
            "max_context": 2000000,
            "cost_per_input_token": 1.25e-06,
            "cost_per_output_token": 1e-05
        },
        "gemini-2.5-flash": {
            "max_context": 1000000,
            "cost_per_input_token": 3e-07,
            "cost_per_output_token": 2.5e-06
        },
        "gemini-1.5-flash": {
            "max_context": 1000000,
            "cost_per_input_token": 75e-08,
            "cost_per_output_token": 3e-07
        },
        "gemini-1.5-flash-8b": {
            "max_context": 1000000,
            "cost_per_input_token": 375e-09,
            "cost_per_output_token": 15e-08
        },
        "gemini-1.5-pro": {
            "max_context": 2000000,
            "cost_per_input_token": 125e-08,
            "cost_per_output_token": 5e-06
        },
        # Will be deprecated from 02/15/2025
        "gemini-1.0-pro": {
            "max_context": 32000,
            "cost_per_input_token": 5e-07,
            "cost_per_output_token": 15e-07
        }
    }

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        genai.configure(api_key=api_key)
        self.model = model
        self.tool_schemas = [{"function_declarations": [self.get_tool_schema(tool) for tool in tools.values()]}]

    @staticmethod
    def get_tool_schema(tool):
        # Gemini function calling will return OpenAPI compatible schema https://ai.google.dev/gemini-api/docs/function-calling
        return {
            "name": tool.NAME,
            "description": tool.DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                "required": list(tool.REQUIRED_PARAMETERS),
            }
        }

    def _call_model(self, system, messages):
        return genai.GenerativeModel(
            model_name=self.model, 
            system_instruction=system).generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                temperature=self.get_param(self.role, "temperature"),
                max_output_tokens=self.get_param(self.role, "max_tokens")
            ),
            tools=self.tool_schemas)

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
                       "parts": str(json.dumps(m.tool_data.result))}
            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": "model" if m.role.value == "assistant" else "user", "parts": "Assistant has no thought!"}
                if m.content is not None and len(m.content) > 0:
                    msg["parts"] = m.content
                if m.tool_data is not None:
                    msg["parts"] = [{"function_call": {
                                        "name": m.tool_data.name,
                                        "args": m.tool_data.arguments
                                    }}]
            else:                
                msg = {"role": "model" if m.role.value == "assistant" else "user", "parts": "Assistant has no thought" if m.content is None else str(m.content)}
            formatted_messages.append(msg)

        try:
            response = self._call_model(system, formatted_messages).to_dict()
            cost = self.calculate_cost(response)
        except ResourceExhausted as e:
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
