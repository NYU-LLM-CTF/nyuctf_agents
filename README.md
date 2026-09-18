# NYU CTF Automation Framework

This repository holds the LLM agents for solving CTF challenges developed for the [NYU CTF Bench](https://nyu-llm-ctf.github.io).
The agents are:

 - **D-CIPHER:** The multi-agent framework involving planner, executor, and auto-prompter agent with enhanced interactions.
 - **NYU CTF Baseline:** The baseline agent presented along with the NYU CTF Bench

The LLM agents operate in a docker environment and interact with CTF challenges to solve them.

## Setup and Installation

The setup requires docker to be installed on the system, please follow instructions for your OS.
The code is tested with atleast python 3.10, earlier versions may work but it is not tested.
It is recommended to create a python virtualenv or conda environment for this setup.

Follow these instructions to setup D-CIPHER or the baseline or both:

1. Clone this repository: `git clone https://github.com/NYU-LLM-CTF/llm_ctf_automation`
2. `cd llm_ctf_automation`
3. Run the setup script (will take a few minutes): `./setup_dcipher.sh` or `./setup_baseline.sh`
    1. The setup script will build the corresponding docker image, and install the python dependencies
    2. You should re-run this setup if the Dockerfile or dependencies are updated
5. Download the NYU CTF dataset (will take a few minutes): `python3 -m nyuctf.download`

## Trial isolation

Each baseline, single-executor, or D-CIPHER evaluation creates a private Docker
bridge and a unique Compose project. Agents and their own challenge services
share that bridge, retaining service names, container-name aliases, and internal
ports. Concurrent runs of the same challenge no longer recreate or stop each
other's containers. Named volumes and additional managed networks are scoped to
the trial. Cleanup runs on setup failure as well as normal exit.

`--network` / `--container-network` now selects the challenge network to replace
(default `ctfnet`), rather than a reusable agent network. A pre-existing `ctfnet`
is no longer needed. Challenges with only an implicit Compose `default` network
use it as the trial network instead. Stop old evaluations before switching versions. Docker
Compose v2 with `config --format json` is required.

Host port publications are removed: evaluations access challenge services by
hostname and internal port. `network_mode` overrides, static addresses/IPAM, unmanaged
external networks, and externally shared named volumes are rejected with an
error rather than silently defeating isolation. Such custom challenge setups
need adaptation. In dataset `v20250206`, `test/2023/CSAW-Finals/web/triathlon_or_sprint`
uses a privileged host-network helper with the Docker socket and is therefore
rejected; it needs a separate challenge-side redesign. Configuration checks accept
the other 156 Compose-backed test/development challenges (this is not a complete
execution of those challenges). Existing challenge bind mounts and internet access are retained;
this change does not isolate shared host files or allowed external services.
For large numbers of concurrent trials, configure Docker address pools to supply
enough independent bridge subnets.

Tests: install `pytest` and the baseline's `ToolDefGenerator` dependency, then run
`PYTHONPATH=. python -m pytest tests -q`. Add `NYUCTF_DOCKER_TESTS=1` for real
Docker/Compose checks using `python:3.12-slim`. These cover both agent frameworks,
file-only and hosted challenges, DNS, internal ports, log/volume cleanup, and
cross-trial access. No model credentials are needed.

## Running D-CIPHER

The main D-CIPHER multi-agent system runs the planner, executor and (optionally) auto-prompter agents.
Use the following command to run it:

```
python3 run_dcipher.py --split <test|development> --challenge <challenge-name> [--enable-autoprompt]
```

To run the ablation experiment of single executor (i.e. without planner), use the following command:

```
python3 run_single_executor.py --split <test|development> --challenge <challenge-name> [--enable-autoprompt]
```

## Running the baseline

Use the following command to run the baseline agent:

```
python3 run_baseline.py -c configs/baseline/base_config.yaml --split <test|development> --challenge <challenge-name>
```

While the baseline agent code is present in the main branch, you can access the baseline's last updated version at [v20250206](https://github.com/NYU-LLM-CTF/llm_ctf_automation/releases/tag/20250206).
This is the code used for the [NYU CTF Bench](https://nyu-llm-ctf.github.io) paper.

