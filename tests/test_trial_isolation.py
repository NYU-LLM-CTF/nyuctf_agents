import io
import json
import os
import shlex
import subprocess
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import docker
import pytest

from nyuctf_multiagent.trial import Trial


def config():
    return {
        "networks": {"ctfnet": {"name": "ctfnet", "external": True}},
        "services": {"web": {"container_name": "challenge", "ports": ["8080:80"],
                              "networks": {"ctfnet": {"aliases": ["web"]}}}},
        "volumes": {"state": {"name": "shared-state"}},
    }


def test_config_preserves_dns_and_scopes_resources():
    challenge = SimpleNamespace(challenge_dir="/tmp/challenge")
    from pathlib import Path
    challenge.challenge_dir = Path(challenge.challenge_dir)
    a, b = Trial(challenge, "ctfnet"), Trial(challenge, "ctfnet")
    with patch.object(Trial, "run", return_value=json.dumps(config())):
        first, second = a.config(), b.config()
    assert first["networks"]["ctfnet"]["name"] != second["networks"]["ctfnet"]["name"]
    assert first["volumes"]["state"]["name"] != second["volumes"]["state"]["name"]
    web = first["services"]["web"]
    assert "container_name" not in web and "ports" not in web
    assert web["networks"]["ctfnet"]["aliases"] == ["web", "challenge"]


def test_implicit_default_challenge_network(tmp_path):
    data = config()
    data["networks"] = {"default": {"name": "old-default"}}
    data["services"]["web"]["networks"] = {"default": None}
    trial = Trial(SimpleNamespace(challenge_dir=tmp_path), "ctfnet")
    with patch.object(trial, "run", return_value=json.dumps(data)):
        result = trial.config()
    assert result["networks"]["default"] == {"name": trial.network, "external": True}
    assert result["services"]["web"]["networks"]["default"]["aliases"] == ["challenge"]


@pytest.mark.parametrize("unsafe", ["host", "external_volume", "static_ip", "missing_network"])
def test_rejects_unsupported_shared_resources_before_start(unsafe, tmp_path):
    data = config()
    if unsafe == "host":
        data["services"]["web"]["network_mode"] = "host"
    elif unsafe == "external_volume":
        data["volumes"]["state"]["external"] = True
    elif unsafe == "static_ip":
        data["services"]["web"]["networks"]["ctfnet"]["ipv4_address"] = "10.1.1.2"
    else:
        data["networks"] = {}
    trial = Trial(SimpleNamespace(challenge_dir=tmp_path, container=True), "ctfnet")
    with patch.object(trial, "run", return_value=json.dumps(data)) as run:
        with pytest.raises(ValueError):
            trial.start()
    assert run.call_count == 1  # Only 'compose config'; no network or service created.


def test_partial_compose_failure_cleans_up(tmp_path):
    trial = Trial(SimpleNamespace(challenge_dir=tmp_path, container=True), "ctfnet")
    error = subprocess.CalledProcessError(1, "up")
    with patch.object(trial, "config", return_value=config()), \
         patch.object(trial, "run") as run, \
         patch.object(trial, "compose", side_effect=[error, "logs", ""]) as compose:
        with pytest.raises(subprocess.CalledProcessError):
            trial.start()
    assert [call.args for call in compose.call_args_list] == [
        ("up", "-d", "--force-recreate"), ("logs",), ("down", "--volumes")]
    assert run.call_args.args == ("docker", "network", "rm", trial.network)
    assert trial.directory is None


@pytest.mark.parametrize("package", ["nyuctf_baseline", "nyuctf_multiagent"])
def test_agent_start_failure_cleans_up_trial(package, tmp_path):
    from importlib import import_module
    environment_type = import_module(package + ".environment").CTFEnvironment
    challenge = SimpleNamespace(category="web", flag="flag{ok}", container=False,
                                files=[], challenge_dir=tmp_path)
    env = environment_type(challenge, "missing-image", "ctfnet")
    with patch.object(env.trial, "run") as run, \
         patch.object(env, "start_docker", side_effect=RuntimeError("agent failed")):
        with pytest.raises(RuntimeError, match="agent failed"):
            env.setup()
    assert run.call_args.args == ("docker", "network", "rm", env.network)


@pytest.mark.skipif(os.getenv("NYUCTF_DOCKER_TESTS") != "1", reason="Docker opt-in")
@pytest.mark.parametrize("package,with_server", [
    ("nyuctf_baseline", False), ("nyuctf_multiagent", False),
    ("nyuctf_baseline", True), ("nyuctf_multiagent", True),
    ("nyuctf_baseline", "default"),
])
def test_real_environment_task_and_peer_isolation(package, with_server, tmp_path):
    from importlib import import_module
    from nyuctf.challenge import CTFChallenge

    environment_type = import_module(package + ".environment").CTFEnvironment
    client = docker.from_env()
    tag = "nyuctf-isolation-test:" + uuid4().hex
    image, _ = client.images.build(
        fileobj=io.BytesIO(b'FROM python:3.12-slim\nCMD ["sleep", "300"]\n'), tag=tag)
    (tmp_path / "challenge.json").write_text(json.dumps({
        "name": "isolation", "category": "web", "flag": "flag{ok}",
        "description": "retrieve the marker", "compose": with_server,
        "box": "challenge", "internal_port": 18080, "files": ["input.txt"],
    }))
    (tmp_path / "input.txt").write_text("task-input")
    network_key = "default" if with_server == "default" else "ctfnet"
    (tmp_path / "docker-compose.yml").write_text(json.dumps({
        "services": {"web": {
            "image": tag, "container_name": "challenge", "working_dir": "/srv",
            "environment": {"LITERAL": "$$literal"},
            "command": ["python", "-m", "http.server", "18080"],
            "ports": ["18080:18080"], "networks": {network_key: {}},
            "volumes": ["state:/srv"],
        }},
        "networks": {network_key: {} if network_key == "default" else {"external": True}},
        "volumes": {"state": {"name": "shared-state"}},
    }))

    def execute(env, command):
        if package == "nyuctf_baseline":
            return env.available_tools["run_command"](command, timeout=10)
        return env.tools["run_command"].call(command, timeout=10)

    def fetch(env, host):
        script = ("import urllib.request; "
                  f"print(urllib.request.urlopen('http://{host}:18080/marker',timeout=1).read().decode())")
        return execute(env, "python -c " + shlex.quote(script))

    networks = []
    try:
        with ExitStack() as stack:
            envs = []
            for _ in range(2):
                challenge = CTFChallenge({"path": ".", "year": "2024", "event": "test"}, tmp_path)
                env = environment_type(challenge, tag, "ctfnet")
                env.setup()
                stack.callback(env.teardown, None, None, None)
                envs.append(env)
                networks.append(env.network)
                assert execute(env, "cat /home/ctfplayer/ctf_files/input.txt")["stdout"].strip() == "task-input"
            a, b = envs
            a_container = client.containers.get(a.container)
            a_container.exec_run(["sh", "-c", "echo private-agent > /tmp/marker"])
            a_container.exec_run(["python", "-m", "http.server", "18080", "--directory", "/tmp"], detach=True)
            for _ in range(30):
                if fetch(a, "127.0.0.1")["returncode"] == 0:
                    break
                time.sleep(.1)
            assert fetch(a, "127.0.0.1")["stdout"].strip() == "private-agent"
            peer_ip = a_container.attrs["NetworkSettings"]["Networks"][a.network]["IPAddress"]
            assert fetch(b, peer_ip)["returncode"] != 0
            assert execute(b, "test ! -e /tmp/marker")["returncode"] == 0
            if with_server:
                servers = []
                for env in envs:
                    server = client.containers.get(env.trial.compose("ps", "-q", "web").strip())
                    marker = uuid4().hex
                    server.exec_run(["sh", "-c", f"echo {marker} > /srv/marker"])
                    assert server.exec_run(["printenv", "LITERAL"]).output == b"$literal\n"
                    assert fetch(env, "challenge")["stdout"].strip() == marker
                    assert not server.attrs["HostConfig"]["PortBindings"]
                    servers.append(server)
                peer_ip = servers[0].attrs["NetworkSettings"]["Networks"][a.network]["IPAddress"]
                assert fetch(b, peer_ip)["returncode"] != 0
                a.teardown(None, None, None)
                assert fetch(b, "challenge")["returncode"] == 0
            # Closing one environment must leave the other usable.
            assert execute(b, "echo alive")["stdout"].strip() == "alive"
        for name in networks:
            with pytest.raises(docker.errors.NotFound):
                client.networks.get(name)
    finally:
        client.images.remove(image.id, force=True)
