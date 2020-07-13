from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from PyPDF2 import PdfFileMerger
from shutil import copyfile
from utils.misc_utils import *
from utils.experiment_utils import *

import json

pdf_filenames = [
    "figs/subbridges/val_all_waymarks_all_bridges_energies.pdf",
    "ais/0/subbridges_ais/mcmc_all_states_all_bridges_energies.pdf",
    "ais/0/subbridges_nuts_post_ais/mcmc_all_states_all_bridges_energies.pdf",
    "figs/val_bridge_logratios.pdf",
    "figs/subbridges/val_all_n_squared_losses.pdf",
    "figs/subbridges/val_norm_of_grad_of_wmark_logp.pdf",
    "figs/subbridges/scatter_noise_vs_ratio_at_data_val.pdf",
    "ais/0/ais_low_to_high_loglik_whole_chains/0_page_0.pdf",
    "ais/0/nuts_post_ais_samples_low_to_high_loglik_whole_chains/0_page_0.pdf",
]

def copy_img_samples(ais_dir, summary_dir, rel_img_dir, config_idx, name):
    src_dir = path_join(ais_dir, rel_img_dir)
    dest_parent_dir = path_join(summary_dir, rel_img_dir)

    for file in os.listdir(src_dir):
        src_file = path_join(src_dir, file)

        dest_dir = path_join(dest_parent_dir, file)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = path_join(dest_dir, "{}_config{}.pdf".format(name, config_idx))

        copyfile(src_file, dest_file)


def collect_ais_files(summary_dir, path, config_idx):
    for j, name in enumerate(["nuts"]):
        ais_dir = path_join(path, "ais/{}/".format(j))
        try:
            copy_img_samples(ais_dir, summary_dir, "ais_low_to_high_loglik_final_states/", config_idx, name)
            copy_img_samples(ais_dir, summary_dir, "ais_low_to_high_loglik_whole_chains/", config_idx, name)
        except Exception as e:
            print("failed to collect ais {} samples. Error: {}".format(j, e))

        try:
            copy_img_samples(ais_dir, summary_dir, "ais_low_to_high_loglik_final_states/", config_idx, name)
            copy_img_samples(ais_dir, summary_dir, "ais_low_to_high_loglik_whole_chains/", config_idx, name)
        except Exception as e:
            print("failed to collect ais {} samples. Error: {}".format(j, e))

        try:
            copy_img_samples(ais_dir, summary_dir, "{}_post_ais_samples_low_to_high_loglik_final_states/".format(name), config_idx, name)
        except Exception as e:
            print("failed to collect post_ais {} samples. Error: {}".format(j, e))


def collect_agg_files(energies_file, losses_file, path, i):
    with open(path_join(path, "config.json")) as f:
        config = AttrDict(json.load(f))

    agg_dir = path_join(path, "aggregated_results/")
    with open(path_join(agg_dir, "val_energies.txt"), "r") as f:
        energies = f.read().split()
        e_str = str(i) + " " + "/".join(energies)
        if "ais_kl" in config: e_str +=  "/" + str(config.ais_kl)
        if "raise_kl" in config: e_str += "/" + str(config.raise_kl)
        e_str += "\n"
        energies_file.write(e_str)

    with open(path_join(agg_dir, "val_losses.txt"), "r") as f:
        losses = f.read().split()
        losses_file.writelines(str(i) + " " + "/".join(losses) + "\n")


def merge_pdfs_for_single_experiment(exp_dir):

    pdfs = ['file1.pdf', 'file2.pdf', 'file3.pdf', 'file4.pdf']
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("result.pdf")
    merger.close()


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Collect files from different experiments"""
    np.set_printoptions(precision=3)

    parser = ArgumentParser(description='Collect files across a set of experiments', formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--config_path', type=str, default="pca_cropped_mnist/20200110-1525")
    # parser.add_argument('--config_path', type=str, default="cropped_mnist/20200114-1441")
    parser.add_argument('--config_path', type=str, default="1d_gauss/20200501-0739")
    parser.add_argument('--id', action='append', type=int, help="model config ids", default=[0])
    args = parser.parse_args()

    summary_dir = path_join(project_root, "saved_models/", args.config_path + "_summary/")
    os.makedirs(summary_dir, exist_ok=True)
    energies_file = open(path_join(summary_dir, "energies.txt"), "w+")
    losses_file = open(path_join(summary_dir, "losses.txt"), "w+")

    for config_idx in args.id:
        path = path_join(project_root, "saved_models/", args.config_path + "_{}".format(config_idx))
        try:
            collect_ais_files(summary_dir, path, config_idx)
        except Exception as e:
            print("failed to collect ais samples for config {}. Error: {}".format(config_idx, e))

        try:
            collect_agg_files(energies_file, losses_file, path, config_idx)
        except Exception as e:
            print("failed to collect agg files for config {}. Error: {}".format(config_idx, e))

        pdf_paths = [path_join(path, fname) for fname in pdf_filenames]
        pdf_paths = [f for f in pdf_paths if os.path.exists(f)]

        merger = PdfFileMerger()
        merger.setPageLayout("/TwoColumnLeft")

        for pdf in pdf_paths:
            merger.append(pdf)

        merger.write(path_join(summary_dir, "{}_results.pdf".format(config_idx)))
        merger.close()

    energies_file.close()
    losses_file.close()


if __name__ == "__main__":
    main()
