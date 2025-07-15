from pathlib import Path
from datetime import datetime
from .config import Config
import getpass
from nyuctf_multiagent.backends import MODELS

class APIKeys(dict):
    """Loads and holds API keys"""
    def __init__(self, key_cfg):
        keys = Path(key_cfg).open("r")
        for line in keys:
            if line.startswith("#"):
                continue
            tag, k = line.strip().split("=")
            self[tag] = k

def load_common_options(parser):
    parser.add_argument("--challenge", required=True, help="Name of the challenge")
    parser.add_argument("--dataset", help="Dataset JSON path. Only provide if not using the NYUCTF dataset at default path")
    parser.add_argument("-n", "--experiment-name", default="default", type=str, help="Experiment name (creates subdir in logdir)")
    parser.add_argument("-s", "--split", default="development", choices=["test", "development"], help="Dataset split to select. Only used when --dataset not provided.")
    parser.add_argument("--keys", default="keys.cfg", help="Path to keys.cfg file for loading API keys")

    parser.add_argument("--container-image", default="ctfenv:multiagent", help="Image tag of docker container")
    parser.add_argument("--container-network", default="ctfnet", help="Network name of docker container")

    # Logging options
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Print debug messages")
    parser.add_argument("-q", "--quiet", default=False, action="store_true", help="Do not print messages to console")
    parser.add_argument("--overwrite-existing", default=False, action="store_true", help="Overwrite existing log")
    parser.add_argument("--skip-existing", default=False, action="store_true", help="Skip if log exists")

def load_config(config_path: str, args) -> Config:
    # TODO this is specific to planner-executor, cleanup later
    config = Config(config_path=config_path)
    if args.planner_model:
        config.planner.model = args.planner_model
    if args.executor_model:
        config.executor.model = args.executor_model
    if args.autoprompter_model:
        config.autoprompter.model = args.autoprompter_model
    if args.max_cost > 0:
        config.experiment.max_cost = args.max_cost

    if config.planner.model not in MODELS:
        raise KeyError(f"Model {config.planner.model} not in options. Select from {', '.join(MODELS.keys())}")
    if config.executor.model not in MODELS:
        raise KeyError(f"Model {config.executor.model} not in options. Select from {', '.join(MODELS.keys())}")
    if config.autoprompter.model not in MODELS:
        raise KeyError(f"Model {config.autoprompter.model} not in options. Select from {', '.join(MODELS.keys())}")

    config.experiment.enable_autoprompt = True if args.enable_autoprompt else config.experiment.enable_autoprompt

    return config

def get_log_filename(args, challenge):
    chalname = challenge.canonical_name
    logdir = Path(args.logdir) / getpass.getuser() / args.experiment_name
    logdir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_existing or args.skip_existing:
        # Keep consistent name if overwriting same or skipping
        return logdir / f"{chalname}.json"
    else:
        # Append datetime to make unique name
        now = datetime.now().strftime("%y%m%d%H%M%S")
        return logdir / f"{chalname}-{now}.json"

class AgentError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
