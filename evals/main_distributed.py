# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pprint
import time

import submitit
import yaml

from evals.scaffold import main as eval_main
from src.utils.logging import get_logger

logger = get_logger(force=True)


try:
    USER = os.getlogin()
except OSError:
    USER = "default"

parser = argparse.ArgumentParser()
parser.add_argument("--val_only", action="store_true", help="only run eval", default=False)
parser.add_argument(
    "--folder",
    type=str,
    help="location to save submitit logs",
    default=f"/home/{USER}/submitit/",
)
parser.add_argument("--override_config_folder", action="store_true")
parser.add_argument("--checkpoint", type=str, help="location of pretrained ckpt")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--nodes", type=int)
parser.add_argument("--exclude", type=str, help="nodes to exclude from training", default=None)
parser.add_argument("--num_segments", type=int, help="Override num_segments (temporal clips)")
parser.add_argument("--num_views", type=int, help="Override num_views_per_segment (spatial views)")
parser.add_argument("--frames_per_clip", type=int, help="Override frames_per_clip")
parser.add_argument(
    "--batch-launch",
    action="store_true",
    help="whether fname points to a file to batch-lauch several config files",
)
parser.add_argument(
    "--fname",
    type=str,
    help="yaml file containing config file names to launch",
    default="configs.yaml",
)
parser.add_argument(
    "--account",
    type=str,
    default="your_account",
    help="Cluster account to use when submitting jobs (replace this placeholder)",
)
parser.add_argument(
    "--partition",
    type=str,
    default="your_partition",
    help="Cluster partition to submit jobs on (replace this placeholder)",
)
parser.add_argument(
    "--qos",
    type=str,
    default="your_qos",
    help="If specified, qos value to use when submitting jobs (replace this placeholder)",
)
parser.add_argument("--time", type=int, default=4300, help="time in minutes to run job")
parser.add_argument("--use_fsdp", action="store_true")


class Trainer:

    def __init__(self, args_eval=None, resume_preempt=None):
        self.eval_name = args_eval["eval_name"]
        self.args_eval = args_eval
        self.resume_preempt = resume_preempt

    def __call__(self):
        eval_name = self.eval_name
        args_eval = self.args_eval
        resume_preempt = self.resume_preempt
        logger.info("loaded eval params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args_eval)

        eval_main(eval_name, args_eval=args_eval, resume_preempt=resume_preempt)

    def checkpoint(self):
        trainer = Trainer(self.args_eval, True)
        return submitit.helpers.DelayedSubmission(
            trainer,
        )


def launch_evals_with_parsed_args(
    args_for_evals,
    submitit_folder,
    account="your_account",
    partition="your_partition",
    qos=None,
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    mem_per_gpu="210G",
    cpus_per_task=32,
    delay_seconds=10,
    exclude_nodes=None,
    save_configs=False,
    dependency=None,
):
    if not isinstance(args_for_evals, list):
        logger.info(f"Passed in eval-args of type {type(args_for_evals)}")
        args_for_evals = [args_for_evals]

    time.sleep(delay_seconds)
    logger.info("Launching evaluations in separate jobs...")
    executor = submitit.AutoExecutor(folder=os.path.join(submitit_folder, "job_%j"), slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_account=account,
        slurm_qos=qos,
        slurm_mem_per_gpu=mem_per_gpu,
        slurm_array_parallelism=64,
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        gpus_per_node=tasks_per_node,
        dependency=dependency,
    )

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    jobs, trainers = [], []
    with executor.batch():
        for ae in args_for_evals:
            trainer = Trainer(ae)
            job = executor.submit(
                trainer,
            )
            trainers.append(trainer)
            jobs.append(job)

    for job, ae in zip(jobs, args_for_evals):
        logger.info(f"Launched eval job with id {job.job_id}")
        if save_configs:
            params_path = os.path.join(job._paths.folder, "eval-params.yaml")
            if not os.path.exists(params_path):
                with open(params_path, "w") as f:
                    yaml.dump(ae, f)
                logger.info(f"Wrote eval config to {params_path}")


def launch_evals():

    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, "r") as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, "r") as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)

            if args.val_only:
                _params["val_only"] = True

            if args.checkpoint:
                _params["model_kwargs"]["checkpoint"] = args.checkpoint

            if args.model_name:
                _params["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = args.model_name

            if args.batch_size:
                _params["experiment"]["optimization"]["batch_size"] = args.batch_size

            if args.nodes:
                _params["nodes"] = args.nodes

            # Apply video classification overrides
            if args.num_segments is not None:
                _params["experiment"]["data"]["num_segments"] = args.num_segments
            if args.num_views is not None:
                _params["experiment"]["data"]["num_views_per_segment"] = args.num_views
            if args.frames_per_clip is not None:
                _params["experiment"]["data"]["frames_per_clip"] = args.frames_per_clip

            # Update tag to reflect overrides
            overrides = []
            if args.num_segments is not None:
                overrides.append(f"seg{args.num_segments}")
            if args.num_views is not None:
                overrides.append(f"view{args.num_views}")
            if args.frames_per_clip is not None:
                overrides.append(f"f{args.frames_per_clip}")

            if overrides:
                override_suffix = "-" + "-".join(overrides)
                if _params.get("tag"):
                    _params["tag"] = _params["tag"] + override_suffix

            if args.override_config_folder:
                _params["folder"] = args.folder
            _params["use_fsdp"] = args.use_fsdp

            nodes = int(_params.get("nodes"))
            tasks_per_node = int(_params.get("tasks_per_node"))
            mem_per_gpu = _params.get("mem_per_gpu", "220G")
            cpus_per_task = _params.get("cpus_per_task", 32)
            configs += [_params]

    logger.info(f"Loaded {len(configs)} config files")
    logger.info(f"Running all jobs with {nodes=} / {tasks_per_node=}")
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Determine submitit folder
    # ---------------------------------------------------------------------- #
    # If --folder was not explicitly provided (still default), use folder from config + /submitit
    default_folder = f"/home/{USER}/submitit/"
    if args.folder == default_folder and len(configs) > 0:
        config_folder = configs[0].get("folder", "")
        if config_folder:
            submitit_folder = os.path.join(config_folder, "submitit")
            logger.info(f"Using submitit folder from config: {submitit_folder}")
        else:
            submitit_folder = args.folder
            logger.info(f"Using default submitit folder: {submitit_folder}")
    else:
        submitit_folder = args.folder
        logger.info(f"Using submitit folder from --folder arg: {submitit_folder}")

    # Ensure submitit folder exists
    os.makedirs(submitit_folder, exist_ok=True)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 4. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_evals_with_parsed_args(
        args_for_evals=configs,
        account=args.account,
        submitit_folder=submitit_folder,
        partition=args.partition,
        qos=args.qos,
        mem_per_gpu=mem_per_gpu,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        exclude_nodes=args.exclude,
    )
    # ---------------------------------------------------------------------- #


if __name__ == "__main__":
    args = parser.parse_args()
    launch_evals()
