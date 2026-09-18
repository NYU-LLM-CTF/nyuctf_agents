import subprocess
import sys
from nyuctf_multiagent.trial import Trial
import json
from pathlib import Path
from nyuctf.challenge import CTFChallenge

from .tools import ToolCall, ToolResult, ALLTOOLS
from .logging import logger

class CTFEnvironment:
    """Manages the docker env for the agent, and the challenge container."""
    def __init__(self, challenge: CTFChallenge, container_image: str, network: str, toolset: str="default"):
        self.challenge = challenge
        self.container_image = container_image
        self.trial = Trial(challenge, network)
        self.network = self.trial.network
        self.container = None
        self.tools = {}
        for tool in ALLTOOLS:
            tool_instance = tool(self)
            self.tools[tool.NAME] = tool_instance

        # The SubmitFlagTool can set this to indicated if flag is found
        self.solved = False
        # The GiveupTool can set this to give up the challenge
        self.giveup = False

    def get_toolset(self, toolset):
        """Return a set of initialized tools"""
        return {name: self.tools[name] for name in toolset}

    def setup(self):
        self.trial.start()
        try:
            self.start_docker()
            for tool in self.tools.values():
                tool.setup()
            for file in self.challenge.files:
                hostpath = self.challenge.challenge_dir / file
                self.copy_into_container(hostpath, f"ctf_files/{file}")
        except BaseException:
            self.teardown(*sys.exc_info())
            raise

    def teardown(self, exc_type, exc_value, traceback):
        try:
            for tool in self.tools.values():
                tool.teardown(exc_type, exc_value, traceback)
        finally:
            try:
                if self.container is not None:
                    self.stop_docker()
                    self.container = None
            finally:
                self.trial.stop()

    def start_docker(self):
        logger.print(f"Starting environment container {self.container_image}...", force=True)
        cmd = ["docker", "run", "-d", "--rm", 
               "--network", self.network, "--platform", "linux/amd64",
               self.container_image]
        output = subprocess.run(cmd, check=True, capture_output=True, text=True)
        self.container = output.stdout.strip()
        logger.debug_message(f"...started {self.container}")

    def copy_into_container(self, hostpath, filename):
        if Path(filename).is_absolute():
            containerpath = Path(filename)
        else:
            containerpath = self.container_home / filename
            # Make parent path (only locals)
            cmd = ["docker", "exec", self.container, "mkdir", "-p", str(containerpath.parent)]
            subprocess.run(cmd, capture_output=True)
        # Copy file
        logger.debug_message(f"Copying file {hostpath} into container {self.container} at {containerpath}")
        cmd = ["docker", "cp", "-aq", str(hostpath), f"{self.container}:{containerpath}"]
        subprocess.run(cmd, capture_output=True)
        return containerpath

    def stop_docker(self):
        logger.print(f"Stopping environment container {self.container_image} {self.container}...", force=True)
        subprocess.run(["docker", "stop", self.container], check=True, capture_output=True)

    def run_tool(self, tool_call):
        # Should have been checked by backend if correct tool or not
        tool = self.tools[tool_call.name]
        res = tool.call(**tool_call.parsed_arguments)
        return ToolResult(name=tool_call.name, id=tool_call.id, result=res)

    @property
    def container_home(self):
        return Path("/home/ctfplayer")

