"""Contains settings that vary per run.

All global, immutable settings should be in settings.py.
"""

import functools
import logging
import subprocess
from argparse import ArgumentParser
from typing import Any, Dict, Optional

import torch

from skill_refactor.settings import CFG, GlobalSettings


def create_arg_parser(
    env_required: bool = True,
    approach_required: bool = True,
    seed_required: bool = True,
) -> ArgumentParser:
    """Defines command line argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--env", required=env_required, type=str)
    parser.add_argument("--approach", required=approach_required, type=str)
    parser.add_argument("--seed", required=seed_required, type=int)
    parser.add_argument("--log_file", default="logs/debug.log", type=str)
    parser.add_argument(
        "--debug_log",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser


@functools.lru_cache(maxsize=None)
def get_git_commit_hash() -> str:
    """Return the hash of the current git commit."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode("ascii").strip()


def update_config_with_parser(parser: ArgumentParser, args: Dict[str, Any]) -> None:
    """Helper function for update_config() that accepts a parser argument."""
    arg_specific_settings = GlobalSettings.get_arg_specific_settings(args)
    # Only override attributes, don't create new ones.
    allowed_args = set(CFG.__dict__) | set(arg_specific_settings)
    # Unfortunately, can't figure out any other way to do this.
    for parser_action in parser._actions:  # pylint: disable=protected-access
        allowed_args.add(parser_action.dest)
    for k in args:
        if k not in allowed_args:
            raise ValueError(f"Unrecognized arg: {k}")
    for k in ("env", "approach", "seed", "experiment_id"):
        if k not in args and hasattr(CFG, k):
            # For env, approach, seed, and experiment_id, if we don't
            # pass in a value and this key is already in the
            # configuration dict, add the current value to args.
            args[k] = getattr(CFG, k)
    for d in [arg_specific_settings, args]:
        for k, v in d.items():
            setattr(CFG, k, v)


def reset_config(
    args: Optional[Dict[str, Any]] = None, default_seed: int = 123
) -> None:
    """Reset to the default CFG, overriding with anything in args.

    This utility is meant for use in testing only.
    """
    parser = create_arg_parser()
    reset_config_with_parser(parser, args, default_seed)


def reset_config_with_parser(
    parser: ArgumentParser,
    args: Optional[Dict[str, Any]] = None,
    default_seed: int = 123,
) -> None:
    """Helper function for reset_config that accepts a parser argument."""
    default_args = parser.parse_args(
        [
            "--env",
            "default env placeholder",
            "--seed",
            str(default_seed),
            "--approach",
            "default approach placeholder",
        ]
    )
    arg_dict = {
        k: v for k, v in GlobalSettings.__dict__.items() if not k.startswith("_")
    }
    arg_dict.update(vars(default_args))
    if args is not None:
        arg_dict.update(args)

    # Force device to CPU if CUDA is not available
    if not torch.cuda.is_available():
        arg_dict["device"] = "cpu"

    update_config_with_parser(parser, arg_dict)


def parse_args(
    env_required: bool = True,
    approach_required: bool = True,
    seed_required: bool = True,
) -> Dict[str, Any]:
    """Parses command line arguments."""
    parser = create_arg_parser(
        env_required=env_required,
        approach_required=approach_required,
        seed_required=seed_required,
    )
    return parse_args_with_parser(parser)


def update_config(args: Dict[str, Any]) -> None:
    """Args is a dictionary of new arguments to add to the config CFG."""
    parser = create_arg_parser()
    update_config_with_parser(parser, args)


def string_to_python_object(value: str) -> Any:
    """Return the Python object corresponding to the given string value."""
    if value in ("None", "none"):
        return None
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value.isdigit() or value.startswith("lambda"):
        return eval(value)
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("["):
        assert value.endswith("]")
        inner_strs = value[1:-1].split(",")
        return [string_to_python_object(s) for s in inner_strs]
    if value.startswith("("):
        assert value.endswith(")")
        inner_strs = value[1:-1].split(",")
        return tuple(string_to_python_object(s) for s in inner_strs)
    return value


def parse_args_with_parser(parser: ArgumentParser) -> Dict[str, Any]:
    """Helper function for parse_args that accepts a parser argument."""
    args, overrides = parser.parse_known_args()
    arg_dict = vars(args)
    if len(overrides) == 0:
        return arg_dict
    # Update initial settings to make sure we're overriding
    # existing flags only
    update_config_with_parser(parser, arg_dict)
    # Override global settings
    assert len(overrides) >= 2
    assert len(overrides) % 2 == 0
    for flag, value in zip(overrides[:-1:2], overrides[1::2]):
        assert flag.startswith("--")
        setting_name = flag[2:]
        if setting_name not in CFG.__dict__:
            raise ValueError(f"Unrecognized flag: {setting_name}")
        arg_dict[setting_name] = string_to_python_object(value)
    return arg_dict
