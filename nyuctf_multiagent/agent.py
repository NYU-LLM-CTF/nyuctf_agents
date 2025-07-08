import time
import json
from pathlib import Path
from nyuctf.challenge import CTFChallenge

from .logging import logger
from .conversation import Conversation, MessageRole, Message
from .tools import DelegateTool, FinishTaskTool, ToolResult, GenAutoPromptTool
from .utils import AgentError

now = lambda: time.time()

class BaseAgent:
    """Base class for an Agent"""
    def __init__(self, environment, challenge, prompter, backend):
        self.environment = environment
        self.challenge = challenge
        self.prompter = prompter
        self.backend = backend

        self.conversation = Conversation()
        self.max_rounds = 30
        self.current_cost = 0.0

    def add_start_prompts(self):
        """
        Adds the system and initial prompts to the conversation.
        This is a separate function to allow adding more params to the prompts.
        """
        self.add_system_message(self.prompter.get("system"))
        self.add_user_message(self.prompter.get("initial"))

    def check_flag_in_response(self, response):
        if response is None:
            return
        stripped_flag = self.challenge.flag
        if "{" in stripped_flag:
            stripped_flag = stripped_flag[stripped_flag.index("{")+1:-1]
        if stripped_flag in response:
            self.environment.solved = True

    # Helper functions to add and print messages to the conversation
    def add_system_message(self, message):
        self.conversation.append_system(message)
        logger.system_message(message)
    def add_user_message(self, message):
        self.conversation.append_user(message)
        logger.user_message(message)
        self.check_flag_in_response(message)
    def add_assistant_message(self, message, tool_call):
        self.conversation.append_assistant(content=message, tool_data=tool_call)
        # Only print thought, action is printed after tool_call is parsed
        logger.assistant_thought(message)
        self.check_flag_in_response(message)
        if tool_call is not None:
            self.check_flag_in_response(tool_call.arguments)

    def add_observation_message(self, tool_result):
        self.conversation.append_observation(tool_data=tool_result)
        # Get truncated output from the conversation
        self.check_flag_in_response(str(self.conversation.all_messages[-1].tool_data.result))

    def run_one_round(self):
        raise NotImplementedError

    def print_parsed_call(self, parsed_call):
        self.environment.tools[parsed_call.name].print_tool_call(parsed_call)
    def print_result(self, tool_result):
        if tool_result.name in self.environment.tools:
            self.environment.tools[tool_result.name].print_result(tool_result)
        else:
            logger.observation_message(tool_result.format())


class SingleAgent(BaseAgent):
    """Single Executor Agent implementation"""
    def __init__(self, environment, challenge, prompter, backend, autoprompter,
                 max_rounds=30, max_cost=1.0, len_observations=5, logfile=None):
        super().__init__(environment, challenge, prompter, backend)
        self.autoprompter = autoprompter
        self.max_rounds = max_rounds
        self.max_cost = max_cost
        self.conversation.len_observations = len_observations
        self.logfile = logfile

    def __enter__(self):
        self.challenge.start_challenge_container()
        self.environment.setup()
        self.start_time = now()
        logger.start_progress()
        return self

    def __exit__(self, ex_type, ex_val, tb):
        self.environment.teardown(ex_type, ex_val, tb)
        self.challenge.stop_challenge_container()
        self.end_time = now()

        error = f"{ex_type.__name__}: {str(ex_val)}" if ex_type is not None else None
        self.dump_log(error=error)
        logger.stop_progress()

    def get_exit_reason(self):
        if self.environment.solved:
            return "solved"
        elif self.environment.giveup:
            return "giveup"
        elif self.total_cost() > self.max_cost:
            return "cost"
        elif self.conversation.round > self.max_rounds:
            return "max_rounds"
        else:
            return "unknown"

    def dump_log(self, error=None):
        if self.logfile is None:
            return

        exit_reason = "error" if error is not None else self.get_exit_reason()
        cost = self.total_cost()
        with self.logfile.open("w") as lf:
            json.dump({
                "start_time": self.start_time,
                "end_time": self.end_time,
                "time_taken": (self.end_time - self.start_time),
                "autoprompter_model": None if not self.autoprompter.enabled else self.autoprompter.backend.model,
                "executor_model": self.backend.model,
                "total_cost": cost,
                "success": self.environment.solved,
                "exit_reason": exit_reason,
                "error": error,
                "autoprompter": [] if not self.autoprompter.enabled else self.autoprompter.conversation.dump(),
                "executor": self.conversation.dump(),
                "debug_log": logger.debug_log,
            }, lf, indent=2)
        if exit_reason == "solved":
            logger.print("[green bold]Challenge Solved![/green bold]", force=True, markup=True)
        else:
            logger.print("[red bold]Challenge Not Solved![/red bold]", force=True, markup=True)
        logger.print(f"exit: {exit_reason} cost: ${cost:.3f} rounds: {self.conversation.round}", force=True)

    def total_cost(self):
        cost = self.current_cost
        if self.autoprompter.enabled:
            cost += self.autoprompter.current_cost
        logger.progress_message(f"${cost:.3f} / ${self.max_cost:.3f}")
        return cost

    def run_one_round(self):
        response = self.backend.send(self.conversation.messages) 
        if response.error is not None:
            raise AgentError(response.error)

        self.current_cost += response.cost
        self.add_assistant_message(response.content, response.tool_call)

        if not response.tool_call:
            self.add_user_message(self.prompter.get("continue"))
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Print unparsed tool_call
            logger.assistant_action(response.tool_call.format())
            # Contains the ToolResult with error
            self.print_result(parsed_call)
            self.add_observation_message(parsed_call)
            return

        # Print parsed tool_call
        self.print_parsed_call(parsed_call)

        tool_result = self.environment.run_tool(parsed_call)
        self.print_result(tool_result)
        self.add_observation_message(tool_result)

    def run_autoprompter(self):
        """Run the autoprompter to set the autoprompt for single agent"""
        # Assumes autoprompter is not None
        while not self.environment.solved and not self.autoprompter.finished \
                and self.autoprompter.conversation.round <= self.autoprompter.max_rounds \
                and self.total_cost() <= self.max_cost:
            self.autoprompter.conversation.next_round()
            self.autoprompter.run_one_round()

        if not self.environment.solved and self.total_cost() <= self.max_cost \
                and self.autoprompter.autoprompt is None:
            # Prompt last time for the autoprompt
            self.autoprompter.run_for_autoprompt()

    def run(self):
        """
        Basic loop to run the agent for fixed number of rounds.
        Calls run_one_round() for each iteration.
        """
        initial_prompt = self.prompter.get("initial")
        if self.autoprompter.enabled:
            # Run the autoprompter if provided
            self.run_autoprompter()
            if self.autoprompter.autoprompt is not None:
                # Only set if autoprompter successfully generates a prompt
                initial_prompt = self.autoprompter.autoprompt
            elif not self.environment.solved:
                logger.print("WARNING! Autoprompter failed to generate a prompt, using the hardcoded one", force=True, style="dark_orange bold")

        logger.print("============= EXECUTOR ===============", style="bold")
        self.add_system_message(self.prompter.get("system"))
        self.add_user_message(initial_prompt)

        while not self.environment.giveup and not self.environment.solved \
                and self.conversation.round <= self.max_rounds \
                and self.total_cost() <= self.max_cost:
            self.conversation.next_round()
            self.run_one_round()


class AutoPromptAgent(BaseAgent):
    """The AutoPrompt will gnerate a prompt and pass it to the Planner-Executor system"""
    def __init__(self, environment, challenge, prompter, backend, max_rounds=10):
        super().__init__(environment, challenge, prompter, backend)
        self.max_rounds = max_rounds
        self.autoprompt = None
        self.finished = False
        self.enabled = False
        self.add_start_prompts()

    def enable_autoprompt(self):
        self.enabled = True

    def run_one_round(self):
        response = self.backend.send(self.conversation.messages)
        if response.error is not None:
            raise AgentError(response.error)
            
        self.current_cost += response.cost
        self.add_assistant_message(response.content, response.tool_call)

        if not response.tool_call:
            self.add_user_message(self.prompter.get("continue"))
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Print unparsed tool_call
            logger.assistant_action(response.tool_call.format())
            # Contains the ToolResult with error
            self.print_result(parsed_call)
            self.add_observation_message(parsed_call)
            return

        # Print parsed tool_call
        self.print_parsed_call(parsed_call)

        if parsed_call.name == GenAutoPromptTool.NAME:
            self.autoprompt = parsed_call.parsed_arguments.get("prompt", None)
            self.finished = True
        else:
            tool_result = self.environment.run_tool(parsed_call)
            self.print_result(tool_result)
            self.add_observation_message(tool_result)

    def run_for_autoprompt(self):
        """
        Prompt the autoprompted last time if it did not already generate a prompt
        """
        self.add_user_message(self.prompter.get("finish_autoprompt"))
        response = self.backend.send(self.conversation.messages)
        self.current_cost += response.cost

        if response.error is not None:
            # Return None if it still errors
            return
        if not response.tool_call:
            # Even if model did not call the tool, we can return any thought content generated.
            self.autoprompt = response.content
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Return unparsed call with content
            logger.assistant_action(response.tool_call.format())
            self.autoprompt = response.content + "\n\n" + response.tool_call.arguments
            return
        # Print parsed tool_call
        self.print_parsed_call(parsed_call)
        if parsed_call.name == GenAutoPromptTool.NAME:
            # Set the task summary
            self.autoprompt = parsed_call.parsed_arguments.get("prompt", None)
        # If any other tool is called, model still does not generate summary.

class PlannerAgent(BaseAgent):
    """The Planner Agent of a multi-agent Planner-Executor system"""
    def __init__(self, environment, challenge, prompter, backend, max_rounds=30):
        super().__init__(environment, challenge, prompter, backend)
        self.max_rounds = max_rounds
        self.delegated_task = None

    def run_one_round(self):
        response = self.backend.send(self.conversation.messages)
        if response.error is not None:
            raise AgentError(response.error)
            
        self.current_cost += response.cost
        self.add_assistant_message(response.content, response.tool_call)

        if not response.tool_call:
            self.add_user_message(self.prompter.get("continue"))
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Print unparsed tool_call
            logger.assistant_action(response.tool_call.format())
            # Contains the ToolResult with error
            self.print_result(parsed_call)
            self.add_observation_message(parsed_call)
            return

        # Print parsed tool_call
        self.print_parsed_call(parsed_call)

        if parsed_call.name == DelegateTool.NAME:
            self.delegated_task = parsed_call
            # MultiAgent system is responsible to add observation to the conversation.
        else:
            tool_result = self.environment.run_tool(parsed_call)
            self.print_result(tool_result)
            self.add_observation_message(tool_result)

class ExecutorAgent(BaseAgent):
    """The Executor Agent of a multi-agent Planner-Executor system"""
    def __init__(self, environment, challenge, prompter, backend, max_rounds=30, len_observations=5):
        super().__init__(environment, challenge, prompter, backend)
        self.max_rounds = max_rounds
        self.conversation.len_observations = len_observations
        self.finished = False
        self.finish_summary = None
        self.error = None

    def new(self):
        """Create new executor with same settings but new conversation"""
        return ExecutorAgent(self.environment, self.challenge, self.prompter,
                             self.backend, max_rounds=self.max_rounds,
                             len_observations=self.conversation.len_observations)

    def run_one_round(self):
        response = self.backend.send(self.conversation.messages)
        if response.error is not None:
            self.finished = True
            self.error = response.error
            # Do not set finish summary
            return

        self.current_cost += response.cost
        self.add_assistant_message(response.content, response.tool_call)

        if not response.tool_call:
            self.add_user_message(self.prompter.get("continue"))
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Print unparsed tool_call
            logger.assistant_action(response.tool_call.format())
            # Contains the ToolResult with error
            self.print_result(parsed_call)
            self.add_observation_message(parsed_call)
            return
        # Print parsed tool_call
        self.print_parsed_call(parsed_call)

        if parsed_call.name == FinishTaskTool.NAME:
            self.finish_summary = parsed_call.parsed_arguments.get("summary", None)
            self.finished = True
            # Executor is done here.
        else:
            tool_result = self.environment.run_tool(parsed_call)
            self.print_result(tool_result)
            self.add_observation_message(tool_result)

    def run_for_finish_summary(self):
        """
        Prompt the executor last time to ask for task summary
        """
        self.add_user_message(self.prompter.get("finish_summary"))
        response = self.backend.send(self.conversation.messages)
        self.current_cost += response.cost

        if response.error is not None:
            # Return None if it still errors
            return
        if not response.tool_call:
            # Even if model did not call the tool, we can return any thought content generated.
            self.finish_summary = response.content
            return

        parsed, parsed_call = self.backend.parse_tool_arguments(response.tool_call)
        if not parsed:
            # Return unparsed call with content
            logger.assistant_action(response.tool_call.format())
            self.finish_summary = response.content + "\n\n" + response.tool_call.arguments
            return
        # Print parsed tool_call
        self.print_parsed_call(parsed_call)
        if parsed_call.name == FinishTaskTool.NAME:
            # Set the task summary
            self.finish_summary = parsed_call.parsed_arguments.get("summary", None)
        # If any other tool is called, model still does not generate summary.

class PlannerExecutorSystem:
    """Holds all the agents of the multi-agent system."""
    def __init__(self, environment, challenge, autoprompter, planner, executor, max_cost=1.0, logfile=None):
        self.environment = environment
        self.challenge = challenge
        self.autoprompter = autoprompter
        self.planner = planner
        self.executor = executor

        self.max_cost = max_cost
        self.logfile = logfile

        self.all_executors = []

    def __enter__(self):
        self.challenge.start_challenge_container()
        self.environment.setup()
        self.start_time = now()
        logger.start_progress()
        return self

    def __exit__(self, ex_type, ex_val, tb):
        self.environment.teardown(ex_type, ex_val, tb)
        self.challenge.stop_challenge_container()
        self.end_time = now()

        error = f"{ex_type.__name__}: {str(ex_val)}" if ex_type is not None else None
        self.dump_log(error=error)
        logger.stop_progress()

    def get_exit_reason(self):
        if self.environment.solved:
            return "solved"
        elif self.environment.giveup:
            return "giveup"
        elif self.total_cost() > self.max_cost:
            return "cost"
        elif self.planner.conversation.round > self.planner.max_rounds:
            return "planner_rounds"
        else:
            return "unknown"

    def dump_log(self, error=None):
        if self.logfile is None:
            return

        exit_reason = "error" if error is not None else self.get_exit_reason()
        cost = self.total_cost()
        with self.logfile.open("w") as lf:
            json.dump({
                "start_time": self.start_time,
                "end_time": self.end_time,
                "time_taken": (self.end_time - self.start_time),
                "autoprompter_model": None if not self.autoprompter.enabled else self.autoprompter.backend.model,
                "planner_model": self.planner.backend.model,
                "executor_model": self.executor.backend.model,
                "total_cost": cost,
                "success": self.environment.solved,
                "exit_reason": exit_reason,
                "error": error,
                "autoprompter": [] if not self.autoprompter.enabled else self.autoprompter.conversation.dump(),
                "planner": self.planner.conversation.dump(),
                "executors": [e.conversation.dump() for e in self.all_executors],
                "executor_errors": [e.error for e in self.all_executors],
                "debug_log": logger.debug_log,
            }, lf, indent=2)
        if exit_reason == "solved":
            logger.print("[green bold]Challenge Solved![/green bold]", force=True, markup=True)
        else:
            logger.print("[red bold]Challenge Not Solved![/red bold]", force=True, markup=True)
        logger.print(f"exit: {exit_reason} cost: ${cost:.3f} planner-rounds: {self.planner.conversation.round} num-executors: {len(self.all_executors)}", force=True)

    def total_cost(self):
        cost = self.planner.current_cost + sum(e.current_cost for e in self.all_executors)
        if self.autoprompter != None:
            cost += self.autoprompter.current_cost
        logger.progress_message(f"${cost:.3f} / ${self.max_cost:.3f}")
        return cost

    def run_autoprompter(self):
        """Run the autoprompter to set the autoprompt for planner"""
        # Assumes autoprompter is not None
        while not self.environment.solved and not self.autoprompter.finished \
                and self.autoprompter.conversation.round <= self.autoprompter.max_rounds \
                and self.total_cost() <= self.max_cost:
            self.autoprompter.conversation.next_round()
            self.autoprompter.run_one_round()

        if not self.environment.solved and self.total_cost() <= self.max_cost \
                and self.autoprompter.autoprompt is None:
            # Prompt last time for the autoprompt
            self.autoprompter.run_for_autoprompt()

    def run(self):
        # Use the hardcoded prompt if no autoprompter
        planner_initial = self.planner.prompter.get("initial")

        if self.autoprompter.enabled:
            # Run the autoprompter if provided
            self.run_autoprompter()
            if self.autoprompter.autoprompt is not None:
                # Only set if autoprompter successfully generates a prompt
                planner_initial = self.autoprompter.autoprompt
            elif not self.environment.solved:
                logger.print("WARNING! Autoprompter failed to generate a prompt, using the hardcoded one", force=True, style="dark_orange bold")

        logger.print("============= PLANNER ===============", style="bold")
        self.planner.add_system_message(self.planner.prompter.get("system"))
        self.planner.add_user_message(planner_initial)

        while not self.environment.solved and not self.environment.giveup and \
                self.planner.conversation.round <= self.planner.max_rounds and \
                self.total_cost() <= self.max_cost:
            self.planner.conversation.next_round()
            self.planner.run_one_round()

            if self.planner.delegated_task is not None:
                result = self.run_executor(self.planner.delegated_task)
                # No need to print this
                tool_result = ToolResult(name=DelegateTool.NAME, id=self.planner.delegated_task.id, result=result)
                self.planner.add_observation_message(tool_result)
                self.planner.delegated_task = None
            
    def run_executor(self, task):
        logger.print("============= EXECUTOR ==============", style="bold")
        executor = self.executor.new() # Create new executor with empty conversation
        self.all_executors.append(executor)

        # Add executor prompts
        task_description = task.parsed_arguments["task"]
        executor.add_system_message(executor.prompter.get("system"))
        executor.add_user_message(executor.prompter.get("initial", task_description=task_description))

        while not self.environment.solved and not executor.finished \
                and executor.conversation.round <= executor.max_rounds \
                and self.total_cost() <= self.max_cost:
            executor.conversation.next_round()
            executor.run_one_round()

        if not self.environment.solved and self.total_cost() <= self.max_cost \
                and executor.finish_summary is None:
            # Prompt last time for finish_summary
            executor.run_for_finish_summary()

        logger.print("============= EXECUTOR DONE =========", style="bold")
        if executor.finished and executor.finish_summary is not None:
            # Send the executor finish summary to the planner.
            return executor.finish_summary
        elif executor.error is not None:
            logger.print(f"Executor Error: {executor.error}", style="red bold")
            return self.executor.prompter.get("finish_error", error=executor.error)
        else:
            # Executor did not complete the task, send empty result
            return self.executor.prompter.get("finish_empty")

