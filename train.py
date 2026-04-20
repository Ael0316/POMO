##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")


##########################################################################################
# import

import argparse
import json
import logging

from utils.utils import create_logger, copy_all_src

from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "train_full.json")

env_params = {}
model_params = {}
optimizer_params = {}
trainer_params = {}
search_params = {}
logger_params = {}


##########################################################################################
# main

def main():
    global env_params, model_params, optimizer_params, trainer_params, search_params, logger_params

    args = _build_parser().parse_args()
    _load_config(args.config_path)
    _apply_runtime_overrides(args)

    if args.debug or DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
        search_params=search_params if search_params else None,
    )

    copy_all_src(trainer.result_folder)

    trainer.run()


def _build_parser():
    parser = argparse.ArgumentParser(description="Train the PO-based TSP model.")
    parser.add_argument(
        "--config_path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training config JSON file.",
    )
    parser.add_argument(
        "--use_cuda",
        default=None,
        help="Optional override for CUDA usage: true/false.",
    )
    parser.add_argument(
        "--cuda_device_num",
        type=int,
        default=None,
        help="Optional override for the CUDA device id.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use the tiny debug training setup.",
    )
    return parser


def _load_config(config_path):
    global env_params, model_params, optimizer_params, trainer_params, search_params, logger_params

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    env_params = config["env_params"]
    model_params = config["model_params"]
    optimizer_params = config["optimizer_params"]
    trainer_params = config["trainer_params"]
    search_params = config.get("search_params", {})
    logger_params = config["logger_params"]


def _parse_optional_bool(value):
    if value is None:
        return None

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _apply_runtime_overrides(args):
    use_cuda = _parse_optional_bool(args.use_cuda)
    if use_cuda is None:
        trainer_params["use_cuda"] = USE_CUDA
    else:
        trainer_params["use_cuda"] = use_cuda

    if args.cuda_device_num is None:
        trainer_params["cuda_device_num"] = CUDA_DEVICE_NUM
    else:
        trainer_params["cuda_device_num"] = args.cuda_device_num


def _set_debug_mode():
    global trainer_params, logger_params
    trainer_params["epochs"] = 2
    trainer_params["train_episodes"] = 10
    trainer_params["train_batch_size"] = 4
    logger_params["log_file"]["desc"] = logger_params["log_file"]["desc"] + "__debug"


def _print_config():
    logger = logging.getLogger("root")
    logger.info("DEBUG_MODE: {}".format(DEBUG_MODE))
    logger.info(
        "USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(
            trainer_params["use_cuda"],
            trainer_params["cuda_device_num"],
        )
    )
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith("params")]


##########################################################################################

if __name__ == "__main__":
    main()
