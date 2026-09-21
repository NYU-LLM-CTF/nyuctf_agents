import time
import subprocess
import os
import json
import openai
import anthropic
from typing import Tuple, Optional, List

from pathlib import Path
from nyuctf.challenge import CTFChallenge

from .ctflogging import status
from .backends import Backend
from .prompts.prompts import PromptManager
from .tools import ToolCall, ToolResult, Tool, TOOLSETS
from .environment import CTFEnvironment

now = lambda: time.time()

class CTFConversation:
    def __init__(self, environment: CTFEnvironment, challenge: CTFChallenge, prompt_manager: PromptManager, backend: Backend, logfile: Path, max_rounds:int=30, max_cost:float=1.0, args=None):
        self.challenge = challenge
        self.environment = environment
        self.prompt_manager = prompt_manager
        self.backend = backend
        self.logfile = logfile

        self.available_functions : dict[str,Tool] = {}

        self.max_rounds = max_rounds
        self.max_cost = max_cost
        # self.config = config
        self.args = args

        self.rounds = 0
        self.cost = 0
        self.finish_reason = "unknown"
        self.model_time = 0
        self.tool_time = 0

    def __enter__(self):
        self.backend.setup()
        self.environment.setup()

        self.start_time = now()
        return self

    def run(self):
        next_msg = self.prompt_manager.initial_message(self.challenge)
        while not self.environment.solved and not self.environment.giveup \
                and self.rounds <= self.max_rounds and self.cost <= self.max_cost:
            try:
                tools_run = self.run_conversation_step(next_msg)
                if tools_run == 0:
                    next_msg = "Please proceed to the next step using your best judgment."
                else:
                    next_msg = None
            except KeyboardInterrupt:
                status.print("[red bold]Interrupted by user[/red bold]", markup=True)
                self.finish_reason = "user_cancel"
                break
            # TODO Normalize the ratelimiterrors
            except (openai.RateLimitError, anthropic.RateLimitError):
                status.print("[red bold]Rate limit reached![/red bold]", markup=True)
                self.finish_reason = "rate_limit"
                break
            except openai.BadRequestError as e:
                msg = str(e)
                if "'code': 'context_length_exceeded'" in msg or "'code': 'string_above_max_length'" in msg:
                    status.print("[red bold]Context length exceeded![/red bold]", markup=True)
                    self.finish_reason = "context_length"
                    break
                else:
                    # Some other error, re-raise
                    raise

        # Look for a finish reason
        if self.environment.solved:
            status.print("[red bold]Challenge solved by our robot overlords![/red bold]", markup=True)
            self.finish_reason = "solved"
        elif self.environment.giveup:
            status.print("[red bold]The LLM decided to give up! NGMI.[/red bold]", markup=True)
            self.finish_reason = "give_up"
        elif self.cost > self.max_cost:
            status.print(f"[red bold]Challenge is unsolved after {self.max_cost} dollars of cost; exiting[/red bold]", markup=True)
            self.finish_reason = "max_cost"
        elif self.rounds > self.max_rounds:
            status.print(f"[red bold]Challenge is unsolved after {self.max_rounds} rounds; exiting[/red bold]", markup=True)
            self.finish_reason = "max_rounds"

    def run_tools(self, tool_calls: List[ToolCall]) -> Tuple[Optional[str],bool]:
        tool_results = []
        for tool_call in tool_calls:
            # Tool lookup
            tool = self.environment.available_tools.get(tool_call.name)
            if not tool:
                status.error_message(f"Unknown tool {tool_call.name}")
                tool_results.append(tool_call.error(f"Unknown tool {tool_call.name}"))
                continue

            # Parse arguments
            parsed, tool_call = self.backend.parse_tool_arguments(tool, tool_call)
            if not parsed:
                tool_results.append(tool_call)
                continue

            try:
                tool_res = tool.run(tool_call)
            except TypeError as e:
                status.debug_message(f"Error encoding results from {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            except Exception as e:
                status.debug_message(f"Error running {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            tool_results.append(tool_res)
        return tool_results

    def run_conversation_step(self, message: Optional[str]=None):
        if message:
            status.user_message(message)
        status.assistant_message("🤔 ...thinking... 🤔")

        # Prompt the model to produce a response and tool_calls
        st = now()
        content, tool_calls, cost = self.backend.send(message)
        self.model_time += now() - st
        self.rounds += 1
        self.cost += cost

        assistant_response = content if content is not None else ""
        for tc in tool_calls:
            assistant_response += f"\n\n```\n{tc.name}: {tc.arguments}\n```"
        if assistant_response:
            status.assistant_message(assistant_response)
        else:
            status.assistant_message("[ no response ]")

        # Run tool calls
        if tool_calls:
            st = now()
            tool_results = self.run_tools(tool_calls)
            self.tool_time += now() - st

            env_response = "## Tool Responses:"
            for tr in tool_results:
                env_response += f"\n\n```\n{tr.name}: {tr.result}\n```\n"
            status.user_message(env_response)
            self.backend.append(tool_results)
            return len(tool_calls)
        else:
            return 0 # No tools run

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = now()
        self.environment.teardown(exc_type, exc_value, traceback)

        self.logfile.write_text(json.dumps(
            {
                "args": vars(self.args),
                "messages": self.backend.get_timestamped_messages(),
                "challenge": self.challenge.challenge_info,
                "solved": self.environment.solved,
                "rounds": self.rounds,
                "cost": self.cost,
                "debug_log": status.debug_log,
                # "challenge_server_output": self.chal.challenge_server_output,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "runtime": {
                    "total": self.end_time - self.start_time,
                    "tools": self.tool_time,
                    "model": self.model_time
                },
                # "exception_info": exception_info,
                "finish_reason": self.finish_reason,
            },
            indent=4
        ))
        status.print(f"Conversation saved to {self.logfile}")
