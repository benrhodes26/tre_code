import json
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.project_constants import ALL_DATASETS
from os.path import join as path_join
from shutil import rmtree
from utils.misc_utils import merge_dicts


def make_script(time, out_err_file_path, command, filename, mem="14000", exclude_teslas=False):

    out_err_dir_path = "/".join(out_err_file_path.split("/")[:-1])
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, "outfiles/{}".format(out_err_dir_path)), exist_ok=True)
    os.makedirs(os.path.join(cwd, "errfiles/{}".format(out_err_dir_path)), exist_ok=True)

    SCRIPT = '#!/bin/sh' \
             '\n#SBATCH -N 1	  # nodes requested' \
             '\n#SBATCH -n 1	  # tasks requested' \
             '\n#SBATCH --gres=gpu:1' \
             '\n#SBATCH --mem={0}  # memory in Mb' \
             '\n#SBATCH --time={1}'.format(mem, time)

    if "8:00:00" in time:
        SCRIPT += '\n#SBATCH --exclude=landonia[01-10]'
    if exclude_teslas and "8:00:00" not in time:
        SCRIPT += '\n#SBATCH --exclude=charles[01-10]'

    SCRIPT += '\n#SBATCH -o outfiles/{0}  # send stdout to outfile' \
              '\n#SBATCH -e errfiles/{0}  # send stderr to errfile' \
              '\nexport CUDA_HOME=/opt/cuda-9.0.176.1/' \
              '\nexport CUDNN_HOME=/opt/cuDNN-7.0/' \
              '\nexport STUDENT_ID=$(whoami)' \
              '\nexport LD_LIBRARY_PATH=${{CUDNN_HOME}}/lib64:${{CUDA_HOME}}/lib64:$LD_LIBRARY_PATH' \
              '\nexport LIBRARY_PATH=${{CUDNN_HOME}}/lib64:$LIBRARY_PATH' \
              '\nexport CPATH=${{CUDNN_HOME}}/include:$CPATH' \
              '\nexport TMP=/disk/scratch/${{STUDENT_ID}}/' \
              '\nexport TMPDIR=/disk/scratch/${{STUDENT_ID}}/' \
              '\nexport PATH=${{CUDA_HOME}}/bin:${{PATH}}' \
              '\nexport PYTHON_PATH=$PATH' \
              '\n# Activate the relevant virtual environment:' \
              '\nsource /home/${{STUDENT_ID}}/miniconda3/bin/activate tensorflow2' \
              '\n{1}'.format(out_err_file_path, command)

    with open(filename, mode='w+') as f:
        f.write(SCRIPT)

    return SCRIPT


def make_waymark_scripts(dataset, time, config_ids=None):
    waymark_scripts = []

    script_dir = make_script_dir(dataset + "/waymark")
    config_dir = path_join(os.getcwd(), "configs/{}/waymark/".format(dataset))
    if not os.path.isdir(config_dir):
        raise FileNotFoundError("Must create a waymark config for dataset {}".format(dataset))

    sorted_ids, config_paths = get_sorted_config_paths(config_dir, config_ids)
    for config_idx, config_path in zip(sorted_ids, config_paths):

        out_err_file_path = "{}/waymark/config{}".format(dataset, config_idx)
        command = 'python create_waymarks.py --config_path={0}/waymark/{1}'.format(dataset, config_idx)
        filename = path_join(script_dir, "{}_con{}.sh".format("_".join(dataset.split("/")), config_idx))

        SCRIPT = make_script(time, out_err_file_path, command, filename)
        waymark_scripts.append(SCRIPT)

    return waymark_scripts


def make_model_scripts(dataset, time, config_ids=None, configs=None, mem="14000", exclude_teslas_for_model=False, restore_ids=None):
    all_scripts = []
    all_save_dirs = []

    script_dir = make_script_dir(dataset + "/model")
    if configs is None:
        config_ids, configs = load_configs(config_ids, dataset, restore_ids)

    for i, c_pair in enumerate(zip(config_ids, configs)):

        config_idx, config = c_pair
        SCRIPT = make_model_script(config_idx, dataset, exclude_teslas_for_model, i, mem, restore_ids, script_dir, time)

        all_scripts.append([SCRIPT])
        all_save_dirs.append(config["save_dir"])

    return all_scripts, all_save_dirs, configs


def make_model_script(config_idx, dataset, exclude_teslas_for_model, i, mem, restore_ids, script_dir, time):
    out_err_file_path = "{}/model/config{}".format(dataset, config_idx)

    if restore_ids:
        command = 'python build_bridges.py --config_path={0}/{1}/config' \
                  ' --restore_model=0 --debug=-1'.format(dataset, restore_ids[i])
    else:
        command = 'python build_bridges.py --config_path={0}/model/{1} --debug=-1'.format(dataset, config_idx)

    filename = path_join(script_dir, "{}_con{}.sh".format("_".join(dataset.split("/")), config_idx))
    SCRIPT = make_script(time, out_err_file_path, command, filename, mem, exclude_teslas=exclude_teslas_for_model)

    return SCRIPT


def make_flow_scripts(dataset, time, config_ids=None, mem="14000", exclude_teslas=False):

    script_dir = make_script_dir(dataset + "/flow/")
    if config_ids is None:
        config_dir = path_join(os.getcwd(), "configs/{}/flow/".format(dataset))
        if not os.path.isdir(config_dir):
            raise FileNotFoundError("must first create flow configs for dataset {}".format(dataset))
        config_ids, _ = get_sorted_config_paths(config_dir)

    scripts = []
    for i in config_ids:

        out_err_file_path = "{}/flow/config{}".format(dataset, i)

        command = 'python train_flow.py --config_path={}/flow/{} --debug=-1'.format(dataset, i)

        filename = path_join(script_dir, "config{}.sh".format(i))
        script = make_script(time, out_err_file_path, command, filename, mem, exclude_teslas=exclude_teslas)
        scripts.append(script)

    return scripts


def load_configs(config_ids, dataset, restore_ids=None):
    if restore_ids is not None:
        config_paths = [path_join(os.getcwd(), "saved_models/{}/{}/config.json".format(dataset, idx)) for idx in restore_ids]
        config_ids = [idx.split("_")[-1] for idx in restore_ids]
    else:
        config_dir = path_join(os.getcwd(), "configs/{}/model/".format(dataset))
        if not os.path.isdir(config_dir):
            raise FileNotFoundError("must first create model configs for dataset {}".format(dataset))
        config_ids, config_paths = get_sorted_config_paths(config_dir, config_ids)

    configs = []
    for path in config_paths:
        with open(path) as f:
            if restore_ids is not None:
                configs.append(json.load(f))
            else:
                config = json.load(f)
                config = merge_dicts(*list(config.values()))
                configs.append(config)

    return config_ids, configs


def make_mi_eval_script(dataset,
                        config_idx,
                        model_idx,
                        script_dir,
                        mem,
                        time,
                        epoch_idx="best",
                        exclude_teslas_for_model=False
                        ):

    out_err_file_path = "{}/post_learning_eval/mi_config{}".format(dataset, config_idx)

    command = 'python representation_learning_evaluation.py ' \
              '--config_path={0}/{1} ' \
              '--eval_epoch_idx={2} ' \
              '--debug=-1'.format(dataset, model_idx, epoch_idx)

    filename = path_join(script_dir, "mi_con{}".format(config_idx))
    SCRIPT = make_script(time=time, out_err_file_path=out_err_file_path, command=command,
                         filename=filename, mem=mem, exclude_teslas=exclude_teslas_for_model and "8:00:00" not in time)
    return SCRIPT


def make_ebm_eval_script(dataset,
                         config_idx,
                         model_idx,
                         script_dir,
                         mem,
                         time,
                         do_sample,
                         do_log_par,
                         id,
                         epoch_idx="best",
                         eval_type="ais",
                         sample_method="hmc",
                         do_post_sample=0,
                         post_ais_n_samples_keep=10,
                         post_ais_thinning_factor=10,
                         only_sample_total_n_steps=1000,
                         only_sample_n_chains=-1,
                         step_size_reduction=3,
                         ais_nuts_max_tree_depth=6,
                         post_ais_nuts_max_tree_depth=10,
                         parallel_iterations=10,
                         swap_memory=-1,
                         do_assess_subbridges=-1,
                         exclude_teslas_for_model=False
                         ):

    out_err_file_path = "{}/post_learning_eval/{}_config{}_id{}".format(dataset, eval_type, config_idx , id)

    command = 'python ebm_evaluation.py ' \
              '--config_path={0}/{1} ' \
              '--do_sample={2} ' \
              '--do_estimate_log_par={3} ' \
              '--do_post_annealed_sample={4} ' \
              '--post_ais_n_samples_keep={5} ' \
              '--post_ais_thinning_factor={6} ' \
              '--ais_id={7} ' \
              '--eval_epoch_idx={8} ' \
              '--sample_method={9} ' \
              '--only_sample_total_n_steps={10} ' \
              '--only_sample_n_chains={11} ' \
              '--step_size_reduction_magnitude={12} ' \
              '--ais_nuts_max_tree_depth={13} ' \
              '--post_ais_nuts_max_tree_depth={14} ' \
              '--parallel_iterations={15} ' \
              '--swap_memory={16} ' \
              '--do_assess_subbridges={17} ' \
              '--debug=-1'.format(dataset,
                                  model_idx,
                                  do_sample,
                                  do_log_par,
                                  do_post_sample,
                                  post_ais_n_samples_keep,
                                  post_ais_thinning_factor,
                                  id,
                                  epoch_idx,
                                  sample_method,
                                  only_sample_total_n_steps,
                                  only_sample_n_chains,
                                  step_size_reduction,
                                  ais_nuts_max_tree_depth,
                                  post_ais_nuts_max_tree_depth,
                                  parallel_iterations,
                                  swap_memory,
                                  do_assess_subbridges)

    filename = path_join(script_dir, "{}_con{}_id{}".format(eval_type, config_idx, id))
    SCRIPT = make_script(time=time, out_err_file_path=out_err_file_path, command=command,
                         filename=filename, mem=mem, exclude_teslas=exclude_teslas_for_model and "8:00:00" not in time)
    return SCRIPT


def make_mi_eval_scripts(shared_args, config):

    eval_scripts = [
        make_mi_eval_script(**shared_args)
        # add more scripts here if desired
    ]

    return eval_scripts


def make_ebm_eval_scripts(shared_args, config):

    eval_scripts = []
    app = lambda x: eval_scripts.append(make_ebm_eval_script(**shared_args, **x))

    do_use_glow = config["noise_dist_name"] == "flow" and config["flow_type"] == "GLOW"
    do_use_glow = 0 if do_use_glow else -1

    app({"id": 0, "do_sample": -1, "do_log_par": 0, "do_post_sample": -1,
         "sample_method": "nuts", "do_assess_subbridges": do_use_glow})

    return  eval_scripts


def make_eval_scripts(model_save_paths, configs, time, exclude_teslas_for_model, mem="14000"):
    all_eval_scripts = []
    for path, config in zip(model_save_paths, configs):

        dataset, model_idx = path.split("/")[-3:-1]
        config_idx = model_idx.split("_")[-1]
        script_dir = make_script_dir(dataset + "/post_learning_eval", False)

        shared_args = {"dataset": dataset, "config_idx": config_idx, "model_idx": model_idx, "script_dir": script_dir,
                       "mem": mem, "time": time, "exclude_teslas_for_model": exclude_teslas_for_model}

        if config["do_mutual_info_estimation"]:
            eval_scripts = make_mi_eval_scripts(shared_args, config)
        else:
            eval_scripts = make_ebm_eval_scripts(shared_args, config)

        all_eval_scripts.append(eval_scripts)

    return all_eval_scripts


def get_sorted_config_paths(config_dir, filter_ids=None):
    rel_config_paths = os.listdir(config_dir)
    sorted_ids = sorted([int(path[:-5]) for path in rel_config_paths])

    if filter_ids is not None:
        sorted_ids = [a for a in sorted_ids if a in filter_ids]

    rel_config_paths = ["{}.json".format(i) for i in sorted_ids]
    config_paths = [path_join(config_dir, p) for p in rel_config_paths]
    return sorted_ids, config_paths


def make_script_dir(dir_name, rm=True):
    cwd = os.getcwd()
    script_dir = path_join(cwd, "scripts/{}".format(dir_name))
    if rm:
        rmtree(script_dir, ignore_errors=True)
    os.makedirs(script_dir, exist_ok=True)
    return script_dir


def main(parse_args=True,
         dataset=None,
         model_config_ids=None,
         flow_config_ids=None,
         model_configs=None,
         model_time=None,
         ais_time=None,
         exclude_teslas=True,
         mem=None,
         restore_ids=None
         ):

    if parse_args:
        parser = ArgumentParser(description='Make slurm scripts for TRE', formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--dataset', type=str, default="multiomniglot")
        parser.add_argument('--model_time', type=str, default="2-00:00:00")
        parser.add_argument('--ais_time', type=str, default="1-00:00:00")
        parser.add_argument('--mem', type=int, default=14000)
        parser.add_argument('--restore_ids', action='append', type=str, help="'timestamp_configid' of models to restore", default=[])

        args = parser.parse_args()
        dataset = args.dataset
        model_time = args.model_time
        ais_time = args.ais_time
        mem = args.mem
        restore_ids = args.restore_ids if args.restore_ids else None

    model_scripts, model_save_dirs, configs = make_model_scripts(dataset, model_time, model_config_ids, model_configs,
                                                                 mem, exclude_teslas, restore_ids)
    eval_scripts = make_eval_scripts(model_save_dirs, configs, ais_time, exclude_teslas, mem)

    flow_scripts = make_flow_scripts(dataset, model_time, flow_config_ids, mem, exclude_teslas)

    return model_scripts, eval_scripts, flow_scripts, model_save_dirs


if __name__ == "__main__":
    main()
