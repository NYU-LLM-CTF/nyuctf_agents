import subprocess
import sys
from pathlib import Path

from nyuctf.challenge import CTFChallenge
from nyuctf_multiagent.trial import Trial

from .ctflogging import status
from .tools import TOOLSETS

class CTFEnvironment:
    """Manages the docker env for the agent, and the challenge container."""
    def __init__(self, challenge: CTFChallenge, container_image: str, network: str):
        self.challenge = challenge
        self.container_image = container_image
        self.trial = Trial(challenge, network)
        self.network = self.trial.network
        self.container = None
        self.available_tools = {}
        for tool in TOOLSETS.get(self.challenge.category, TOOLSETS['default']):
            tool_instance = tool(self)
            self.available_tools[tool_instance.name] = tool_instance

        # The CheckFlag tool can set this to indicated if flag is found
        self.solved = False
        # The GiveUp tool can set this to give up the challenge
        self.giveup = False

    def setup(self):
        self.trial.start()
        try:
            self.start_docker()
            self.trial.configure_agent(self.container)
            for tool in self.available_tools.values():
                tool.setup()
            for file in self.challenge.files:
                hostpath = self.challenge.challenge_dir / file
                self.copy_into_container(hostpath, f"ctf_files/{file}")
        except BaseException:
            self.teardown(*sys.exc_info())
            raise

    def teardown(self, exc_type, exc_value, traceback):
        try:
            for tool in self.available_tools.values():
                tool.teardown(exc_type, exc_value, traceback)
        finally:
            try:
                if self.container is not None:
                    self.stop_docker()
                    self.container = None
            finally:
                self.trial.stop()

    def start_docker(self):
        status.debug_message(f"Starting environment container {self.container_image}...")
        cmd = ["docker", "run", "-d", "--rm", 
               "--network", self.network, "--platform", "linux/amd64",
               self.container_image]
        output = subprocess.run(cmd, check=True, capture_output=True, text=True)
        self.container = output.stdout.strip()
        status.debug_message(f"...started {self.container}")

    def copy_into_container(self, hostpath, filename):
        if Path(filename).is_absolute():
            containerpath = Path(filename)
        else:
            containerpath = self.container_home / filename
            # Make parent path (only locals)
            cmd = ["docker", "exec", self.container, "mkdir", "-p", str(containerpath.parent)]
            subprocess.run(cmd, capture_output=True)
        # Copy file
        status.debug_message(f"Copying file {hostpath} into container {self.container} at {containerpath}")
        cmd = ["docker", "cp", "-aq", str(hostpath), f"{self.container}:{containerpath}"]
        subprocess.run(cmd, capture_output=True)
        return containerpath

    def stop_docker(self):
        status.debug_message(f"Stopping environment container {self.container_image} {self.container}...")
        subprocess.run(["docker", "stop", self.container], check=True, capture_output=True)

    @property
    def container_home(self):
        return Path("/home/ctfplayer")
