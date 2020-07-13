import tensorflow_datasets as tfds
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from __init__ import density_data_root
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


def import_omniglot_dataset():
    data_dict = tfds.load("omniglot", batch_size=5000)
    tf_train = data_dict["train"]

    train_iterator = tf.compat.v1.data.make_one_shot_iterator(tf_train)
    batch = train_iterator.get_next()
    img_batch = batch["image"][:, :, :, :1]
    resized_img_batch = tf.image.resize(img_batch, tf.constant([28, 28], dtype=tf.int32))

    alphabet_idxs = batch["alphabet"]
    alphabet_char_idxs = batch["alphabet_char_id"]
    labels = batch["label"]

    with tf.Session() as sess:
        data, alph_idxs, alph_char_idxs, l = \
            tf_batched_operation(sess,
                                 [resized_img_batch, alphabet_idxs, alphabet_char_idxs, labels],
                                 19280,
                                 5000,
                                 const_feed_dict={}
                                 )

    omniglot_dir = density_data_root + "omniglot/"
    os.makedirs(omniglot_dir, exist_ok=True)
    np.savez_compressed(
        omniglot_dir + "data",
        train=data,
        alphabet_idxs=alph_idxs,
        alphabet_char_idxs=alph_char_idxs,
        labels=l
    )


def create_multiomniglot(do_plot=False):
    loaded = np.load(density_data_root + "omniglot/data.npz")
    data = loaded["train"]
    alph_idxs = loaded["alphabet_idxs"]
    alph_char_idxs = loaded["alphabet_char_idxs"]
    n_imgs_per_character = 20
    n_trn = 50000
    n_val = 10000
    n_tst = 10000

    alphabets = sort_into_alphabets(alph_char_idxs, alph_idxs, data, do_plot)

    data_path = path_join(density_data_root, "omniglot")
    os.makedirs(data_path, exist_ok=True)
    for d in [1, 4, 9]:

        dataset, labels = [], []
        for i in range(n_trn + n_val + n_tst):

            alphabet_sizes = np.array([len(A) for A in alphabets[:d]])
            z = np.random.randint(low=np.zeros(d), high=alphabet_sizes, size=d)

            x_img_idxs = np.random.randint(low=np.zeros(d), high=n_imgs_per_character, size=d)
            y_img_idxs = np.random.randint(low=np.zeros(d), high=n_imgs_per_character, size=d)

            x_imgs = [alphabets[i][z[i]][x_img_idxs[i]] for i in range(d)]
            y_imgs = [alphabets[i][(z[i] + 1) % alphabet_sizes[i]][y_img_idxs[i]] for i in range(d)]

            x = np.concatenate(x_imgs, axis=-1)
            y = np.concatenate(y_imgs, axis=-1)

            xy = np.stack([x, y], axis=-1)
            dataset.append(xy)
            labels.append(z)

            # plot_multiomniglot_datapoint(xy, i, d)

        dataset, labels = np.array(dataset), np.array(labels)

        np.savez_compressed(path_join(data_path, "multiomniglot_trn_{}".format(d)),
                            data=dataset[:n_trn], labels=labels[:n_trn])
        np.savez_compressed(path_join(data_path, "multiomniglot_val_{}".format(d)),
                            data=dataset[n_trn:n_trn+n_val], labels=labels[n_trn:n_trn+n_val])
        np.savez_compressed(path_join(data_path, "multiomniglot_tst_{}".format(d)),
                            data=dataset[n_trn + n_val:], labels=labels[n_trn + n_val:])


def plot_multiomniglot_datapoint(xy, i, d):
    fig_dir = project_root + "figs/omniglot/multiomniglot/"
    os.makedirs(fig_dir, exist_ok=True)
    n_rows = n_cols = int(d ** 0.5)
    fig, axs = plt.subplots(2 * n_rows, n_cols, figsize=(n_cols, 2 * n_rows))
    axs = axs.ravel()
    imgs = xy.reshape((28, 28, 2 * d))
    for j in range(2 * d):
        img = imgs[:, :, j]
        axs[j].imshow(img, cmap="gray")
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
        axs[j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    save_fig(fig_dir, "datapoint_{}".format(i), bbox_inches='tight')


def sort_into_alphabets(alph_char_idxs, alph_idxs, data, do_plot):
    """construct list of alphabets [A_1, ..., A_max]
    where alphabet A_i contains N_i characters and Ai = [C1, ..., C_{N_i}]
    where Cj is an array containing images that represent the jth character of A_i
    """
    alphabets = []
    for i in np.unique(alph_idxs):
        alph_gather_idxs = np.nonzero(alph_idxs == i)
        alphabet_i = data[alph_gather_idxs]
        char_idxs_for_alphabet_i = alph_char_idxs[alph_gather_idxs]

        characters_for_alphabet_i = []
        for j in np.sort(np.unique(char_idxs_for_alphabet_i)):
            char_j_gather_idxs = np.nonzero(char_idxs_for_alphabet_i == j)
            character_ij = alphabet_i[
                char_j_gather_idxs]  # array of images that represent jth character of ith alphabet
            characters_for_alphabet_i.append(character_ij)

        alphabets.append(characters_for_alphabet_i)

    # sort in decreasing order of alphabet size
    alphabets.sort(reverse=True, key=lambda x: len(x))

    # plot the alphabets so I can spot check for correctness (alphabet 19 is english)
    if do_plot:
        plot_all_omniglot_alphabets(alphabets)

    return alphabets


def plot_all_omniglot_alphabets(alphabets):
    fig_dir = project_root + "figs/omniglot/alphabets/"
    os.makedirs(fig_dir, exist_ok=True)
    for i, A in enumerate(alphabets):
        fig, axs = plt.subplots(len(A), len(A[0]), figsize=(len(A[0]), len(A)))
        axs = axs.ravel()
        ax_num = 0
        for C in A:
            for c in C:
                axs[ax_num].imshow(np.squeeze(c), cmap="gray")
                axs[ax_num].set_xticklabels([])
                axs[ax_num].set_yticklabels([])
                axs[ax_num].axis('off')
                ax_num += 1

        plt.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        save_fig(fig_dir, "omniglot_alphabet_{}".format(i), bbox_inches='tight')


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Run density estimation experiment with sequential noise-contrastive estimation"""
    make_logger()
    logger = logging.getLogger("tf")
    np.set_printoptions(precision=3)

    import_omniglot_dataset()
    create_multiomniglot()


if __name__ == "__main__":
    main()
