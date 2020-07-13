import json
import os
import subprocess

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from __init__ import project_root
from os.path import join as path_join
from time import sleep
from make_slurm_scripts import main as make_slurm_scripts_main
from make_slurm_scripts import get_sorted_config_paths
from utils.misc_utils import merge_dicts
from utils.experiment_utils import *


def get_config_ids(dataset, arg_ids, restore_ids=None, train_flow=False):
    if restore_ids is not None:
        config_paths = [path_join(os.getcwd(), "saved_models/{}/{}/config.json".format(dataset, idx)) for idx in restore_ids]
        model_config_ids = [idx.split("_")[-1] for idx in restore_ids]
    else:
        model_config_dir = path_join(project_root, "configs/{}/model/".format(dataset))
        model_config_ids, config_paths = get_sorted_config_paths(model_config_dir, arg_ids)

    if train_flow:
        flow_config_dir = path_join(project_root, "configs/{}/flow/".format(dataset))
        flow_config_ids, _ = get_sorted_config_paths(flow_config_dir, arg_ids)
    else:
        flow_config_ids = None

    # load config files for models
    model_configs = []
    for config_idx, config_path in zip(model_config_ids, config_paths):
        with open(config_path) as f:
            if restore_ids is not None:
                config = json.load(f)
                model_configs.append(config)
            else:
                config = json.load(f)
                config = merge_dicts(*list(config.values()))
                model_configs.append(config)

    return model_configs, model_config_ids, flow_config_ids


# noinspection PyUnresolvedReferences
def try_submitting_model_scripts(i, unrun_model_jobs, use_apollo=False):
    conf, model_scripts_for_one_config = unrun_model_jobs[i]

    logger = logging.getLogger("tf")
    logger.info("Submitting model scripts!")
    for script in model_scripts_for_one_config:
        if debug_mode:
            subprocess.run(["echo", script])
        else:
            commands = ["sbatch"] if not use_apollo else ["sbatch", "-p", "apollo"]
            subprocess.run(commands, input=script, universal_newlines=True)

    unrun_model_jobs.pop(i)


# noinspection PyUnresolvedReferences
def try_submitting_eval_scripts(i, unrun_eval_jobs):

    model_conf, model_dir, eval_scripts_for_one_model = unrun_eval_jobs[i]

    if os.path.isdir(model_dir):

        all_jobs_finished = check_all_jobs_finished(model_dir, n_jobs=1)  # check all jobs for one model have completed

        if all_jobs_finished:
            logger = logging.getLogger("tf")
            logger.info("Submitting eval scripts!")
            if debug_mode:
                for e_script in eval_scripts_for_one_model:
                    subprocess.run(["echo", e_script])
            else:
                for e_script in eval_scripts_for_one_model:
                    subprocess.run(["sbatch"], input=e_script, universal_newlines=True)
                    sleep(20)

            unrun_eval_jobs.pop(i)


def check_all_jobs_finished(model_dir, n_jobs):
    all_jobs_finished = True
    for j in range(n_jobs):
        job_dir = os.path.join(model_dir, "job_{}".format(j))

        if not os.path.isdir(job_dir):
            all_jobs_finished = False
            break
        elif not os.path.isfile(os.path.join(job_dir, "finished.txt")):
            all_jobs_finished = False
            break
    return all_jobs_finished


def wait_until_scripts_finish(model_save_dirs, sleep_time):
    scripts_still_running = True
    while scripts_still_running:

        all_finished = True
        for save_dir in model_save_dirs:
            ais_dir = os.path.join(save_dir, "ais/0/")
            if not os.path.isdir(ais_dir):
                all_finished = False
                break
            elif not os.path.isfile(os.path.join(ais_dir, "finished.txt")):
                all_finished = False
                break

        scripts_still_running = not all_finished

        logger = logging.getLogger("tf")
        logger.info("Scripts still haven't finished running. Resting...")
        sleep(sleep_time)


def get_slurm_scripts(args, model_config_ids, model_configs, restore_ids=None, flow_config_ids=None):
    all_model_scripts, eval_scripts, flow_scripts, model_save_dirs = \
        make_slurm_scripts_main(parse_args=False,
                                dataset=args.dataset,
                                model_config_ids=model_config_ids,
                                flow_config_ids=flow_config_ids,
                                model_configs=model_configs,
                                model_time=args.model_time,
                                ais_time=args.ais_time,
                                exclude_teslas=args.exclude_teslas_for_model != -1,
                                mem=args.mem,
                                restore_ids=restore_ids)

    return all_model_scripts, eval_scripts, flow_scripts, model_save_dirs


def run_aggregate(args, model_config_ids, model_timestamp):
    for i in model_config_ids:
        subprocess.run(["python", "aggregate_results.py",
                        "--config_path={}/{}_{}".format(args.dataset, model_timestamp, i)
                        ])


# noinspection PyUnresolvedReferences
def main():
    make_logger()
    logger = logging.getLogger("tf")

    parser = ArgumentParser(description='Run entire TRE pipeline for a particular dataset and configs',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', action='append', type=int, help="model config ids", default=[])
    parser.add_argument('--restore_timestamp', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="multiomniglot")
    parser.add_argument('--model_time', type=str, default="2-00:00:00")
    parser.add_argument('--ais_time', type=str, default="1-00:00:00")
    parser.add_argument('--mem', type=int, default=14000)
    parser.add_argument('--exclude_teslas_for_model', type=int, default=-1)  # -1 == False, else True
    parser.add_argument('--use_apollo', type=int, default=-1)  # -1 == False, else True
    parser.add_argument('--train_flow', type=int, default=-1)  # -1 == False, otherwise True
    parser.add_argument('--only_eval', type=int, default=-1)  # -1 == False, otherwise True
    parser.add_argument('--only_plot_comparisons', type=int, default=-1)  # -1 == False, otherwise True
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--extra_id', action='append', type=str,
                        help="timestamps_id of extra models to include in comparison plot", default=[])
    parser.add_argument('--debug_mode', type=int, default=-1)  # -1 == False
    args = parser.parse_args()

    if args.restore_timestamp:
        restore_ids = ["{}_{}".format(args.restore_timestamp, idx) for idx in args.id]
    else:
        restore_ids = None
    only_eval = args.only_eval != -1
    train_flow = args.train_flow != -1

    globals()["debug_mode"] = False if args.debug_mode == -1 else True
    sleep_time = 5 if debug_mode else 60

    # load model/flow configs
    model_configs, model_config_ids, flow_config_ids\
        = get_config_ids(args.dataset, args.id, restore_ids, train_flow)

    # create slurm scripts
    all_model_scripts, eval_scripts, flow_scripts, model_save_dirs = \
        get_slurm_scripts(args, model_config_ids, model_configs, restore_ids, flow_config_ids)

    if train_flow:
        for flow_script in flow_scripts:
            if debug_mode:
                subprocess.run(["echo", flow_script])
            else:
                subprocess.run(["sbatch"], input=flow_script, universal_newlines=True)

    else:
        model_timestamp = model_save_dirs[0].split("/")[-2].split("_")[-2]  # awful

        # keep trying to submit model and eval scripts until they are all done
        unrun_model_jobs = list(zip(model_configs, all_model_scripts))
        unrun_eval_jobs = list(zip(model_configs, model_save_dirs, eval_scripts))

        while (len(unrun_model_jobs) > 0) or (len(unrun_eval_jobs) > 0):

            if only_eval:
                unrun_model_jobs = []  # don't train any models, just run eval
            else:
                for i in range(len(unrun_model_jobs)-1, -1, -1):  # loop backwards
                    try_submitting_model_scripts(i, unrun_model_jobs, args.use_apollo != -1)

            for i in range(len(unrun_eval_jobs)-1, -1, -1):  # loop backwards
                try_submitting_eval_scripts(i, unrun_eval_jobs)

            logger.info("Still haven't run all scripts. Resting...")
            sleep(sleep_time)  # take a rest

        logger.info("Finished submitting all scripts!")

        run_aggregate(args, model_config_ids, model_timestamp)
        wait_until_scripts_finish(model_save_dirs, sleep_time)

        id_args = ["--id={}_{}".format(model_timestamp, i) for i in model_config_ids]
        id_args += ["--id={}".format(id) for id in args.extra_id]

        # When eval scripts all finish, run plotting script to compare all models
        subprocess.run(["python", "plot_model_comparisons.py",
                        "--dataset={}".format(args.dataset),
                        # "--model_timestamp={}".format(model_timestamp),
                        *id_args,
                        "--experiment_name={}".format(args.experiment_name)])

        logger.info("Finished!!")


if __name__ == "__main__":
    main()
