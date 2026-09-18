"""Docker resources owned by one baseline or multiagent evaluation."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class Trial:
    def __init__(self, challenge, network):
        self.challenge = challenge
        self.base_network = network
        self.project = "nyuctf-" + uuid4().hex
        self.network = self.project + "-agent"
        self.directory = None
        self.network_created = False
        self.compose_started = False

    @staticmethod
    def run(*args):
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout

    def compose(self, *args):
        return self.run("docker", "compose", "--project-name", self.project,
                        "-f", str(Path(self.directory.name) / "compose.json"), *args)

    def config(self):
        # Compose resolves relative build and bind paths against the original file.
        data = json.loads(self.run(
            "docker", "compose", "--project-name", self.project,
            "-f", str(self.challenge.challenge_dir / "docker-compose.yml"),
            "config", "--format", "json"))
        networks = data.setdefault("networks", {})
        agent_keys = [key for key, value in networks.items()
                      if key == self.base_network or value.get("name") == self.base_network]
        # Some packaged challenges use only Compose's implicit default network.
        if not agent_keys and set(networks) == {"default"}:
            agent_keys = ["default"]
        if not agent_keys:
            raise ValueError(f"Challenge has no {self.base_network!r} network")
        for key, value in networks.items():
            if value.get("driver", "bridge") != "bridge" or value.get("ipam"):
                raise ValueError("Trial isolation requires bridge networks without static IPAM")
            if key in agent_keys:
                networks[key] = {"name": self.network, "external": True}
            else:
                if value.get("external"):
                    raise ValueError(f"Unmanaged external challenge network: {key}")
                value["name"] = f"{self.project}-{key}"
        for service in data["services"].values():
            if "network_mode" in service:
                raise ValueError("network_mode bypasses trial network isolation")
            # Host-published ports bypass bridge isolation; agents use internal ports.
            service.pop("ports", None)
            name = service.pop("container_name", None)
            for key, endpoint in service.get("networks", {}).items():
                endpoint = endpoint or {}
                if any(k in endpoint for k in ("ipv4_address", "ipv6_address")):
                    raise ValueError("Static service addresses are not trial-safe")
                if name:
                    endpoint.setdefault("aliases", []).append(name)
                service["networks"][key] = endpoint
        for key, volume in data.get("volumes", {}).items():
            if volume.get("external") or volume.get("driver_opts"):
                raise ValueError(f"Unmanaged shared challenge volume: {key}")
            volume["name"] = f"{self.project}-{key}"
        return data

    def start(self):
        try:
            # Validate the challenge before allocating any Docker resources.
            config = self.config() if self.challenge.container else None
            self.run("docker", "network", "create", "--driver", "bridge", self.network)
            self.network_created = True
            if config is not None:
                self.directory = tempfile.TemporaryDirectory(prefix=self.project)
                # Compose's serialized config already escapes literal dollar signs.
                (Path(self.directory.name) / "compose.json").write_text(
                    json.dumps(config))
                self.compose_started = True  # 'up' can create resources before failing.
                self.compose("up", "-d", "--force-recreate")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        # Attempt every cleanup even when an earlier operation fails.
        if self.compose_started:
            try:
                self.challenge.container_output = self.compose("logs")
            except Exception:
                logger.exception("Could not collect challenge logs for %s", self.project)
            try:
                self.compose("down", "--volumes")
                self.compose_started = False
            except Exception:
                logger.exception("Could not stop challenge %s", self.project)
        if self.network_created:
            try:
                self.run("docker", "network", "rm", self.network)
                self.network_created = False
            except Exception:
                logger.exception("Could not remove trial network %s", self.network)
        if self.directory is not None and not self.compose_started:
            self.directory.cleanup()
            self.directory = None
