# -*- coding: utf-8 -*-
"""Data providers.
This module provides classes for loading datasets and iterating over batches of
data points.
"""
import data_handlers
import numpy as np

DEFAULT_SEED = 22012018


class DataProvider(object):
    """Data provider that can be iterated over to obtain minibatches of data."""

    def __init__(self,
                 data,
                 batch_size,
                 max_num_batches=-1,
                 labels=None,
                 use_labels=None,
                 shuffle_order=True,
                 shuffle_waymarks=False,
                 seed=None):
        """Create a new data provider object.
        Args:
            data (ndarray): Array of data input features of shape
                (num_data, input_dim).
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            seed (int): A seed for numpy random number generator.
        """
        self.data = data

        self.use_labels = use_labels
        self.labels = labels

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order

        # If data contains a waymark dimension, then we may also shuffle the second dimension
        # Note, however, that this shuffle works differently to the shuffling of the datapoints.
        # For *each* slice into the waymark dimension, we apply a *different* permutation
        self.shuffle_waymarks = shuffle_waymarks
        self._current_data_order = np.arange(data.shape[0])
        if seed is None:
            seed = DEFAULT_SEED
        self.seed = seed
        self.data_rng = np.random.RandomState(seed)
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""

        possible_num_batches = self.data.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.
        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        if not self.shuffle_waymarks:
            data_inv_perm = np.argsort(self._current_data_order)
            self._current_data_order = self._current_data_order[data_inv_perm]
            self.data = self.data[data_inv_perm]
            if self.use_labels:
                self.labels = self.labels[data_inv_perm]
        else:
            raise NotImplementedError("Cannot reset data provider when self.shuffle_waymarks == True")

    def shuffle(self):
        """Randomly shuffles order of data"""
        if not self.shuffle_waymarks:
            data_perm = self.data_rng.permutation(self.data.shape[0])
            self._current_data_order = self._current_data_order[data_perm]
            self.data = self.data[data_perm]
            if self.use_labels: self.labels = self.labels[data_perm]
        else:
            self.data = self.independent_sliced_shuffle(self.data)

    @staticmethod
    def independent_sliced_shuffle(data):
        """Given a matrix 'data' of shape [n, k, ...], apply a different permumation
           along axis 0 of each slice data[:, i, ...]
        """
        n, k = data.shape[:2]
        other_dims = data.shape[2:]
        assert len(other_dims) in [1, 3], "expected either 1 or 3 event dims, but got {}".format(len(other_dims))
        data = data.reshape(n, k, -1)
        d = data.shape[-1]

        # data has shape (n, num_waymarks, ...). Shuffle each of the num_waymarks columns *independently*.
        idx = np.random.rand(*data.shape[:2]).argsort(0)
        idx = np.repeat(idx.reshape(-1, 1), d, axis=-1).reshape(n, k, d)
        idx2 = np.repeat(np.arange(k), d, axis=-1).reshape(k, -1)
        idx2 = np.tile(idx2, [n, 1]).reshape(n, k, d)
        idx3 = np.repeat(np.arange(d)[:, np.newaxis], k, axis=-1).T
        idx3 = np.tile(idx3, [n, 1]).reshape(n, k, d)
        data = data[idx, idx2, idx3]

        return data.reshape(n, k, *other_dims)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end.

        If self.noise is not None, then also return a noise batch"""
        if self._curr_batch + 1 > self.num_batches:
            self.new_epoch()
            raise StopIteration()

        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size, (self._curr_batch + 1) * self.batch_size)
        ret = self.data[batch_slice]
        if self.use_labels:
            ret = (ret, self.labels[batch_slice])

        self._curr_batch += 1
        return ret

    def get_rand_batch(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        n = self.data.shape[0]
        start = np.random.randint(n - batch_size)
        ret = self.data[start: start + batch_size]
        if self.use_labels:
            ret = (ret, self.labels[start: start + batch_size])

        return ret

    def __repr__(self):
        return "DataProvider"


# noinspection PyMissingConstructor
class DataProviderFromSource(DataProvider):

    def __init__(self,
                 data_source,
                 batch_size,
                 use_labels=False,
                 mode="trn",
                 frac=1.0,
                 max_num_batches=-1,
                 shuffle_order=True,
                 shuffle_waymarks=False,
                 seed=None):
        """Create a new data provider object.
        Args:
            data_source: object containing all data (train, val & test)
            dataset_name (string): see load_data method for options.
            batch_size (int): Number of data points to include in each batch.
            mode (string: either "trn", "val" or "tst"
            frac (float): fraction of data to use (useful for fast debugging)
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            seed (int): Seed of random number generator.
        """
        self.mode = mode  # e.g train/val/tst
        self.source = source = getattr(data_source, mode)  # object containing data, metadata & preprocessing operations
        if "2d" in repr(data_source) or "1d" in repr(data_source):
            self.source_1d_or_2d = data_source

        self.n_samples = self.data.shape[0]
        n = max(1000, int(self.n_samples * frac))
        self.data = self.data[: n]  # potentially use a fraction of data (for faster debugging)
        self.n_dims = np.product(self.data.shape[1:])
        self.labels = getattr(source, "labels", None)
        self.use_labels = use_labels

        self.ldj = getattr(source, "ldj", None)  # log det jacobian of preprocessing

        if batch_size < 1: raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1: raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()

        self.shuffle_order = shuffle_order
        self.shuffle_waymarks = shuffle_waymarks
        self._current_data_order = np.arange(self.data.shape[0])

        if seed is None: seed = DEFAULT_SEED
        self.seed = seed
        self.data_rng = np.random.RandomState(seed)

        self.new_epoch()

    @property
    def data(self):
        """Number of data points to include in each batch."""
        return self.source.x

    @data.setter
    def data(self, value):
        self.source.x = value

    def get_samples_for_label(self, class_idx):
        assert not self.shuffle_order, "datapoints have been shuffled, so in different order to labels"
        return self.data[self.labels == class_idx]

    def __repr__(self):
        return "DensityEstimationDataProvider"


def load_data_providers(dataset_name,
                        batch_size,
                        seed,
                        use_labels=False,
                        frac=1,
                        shuffle=True,
                        data_args=None):
    """
    generate the train, val and test set data providers for the dataset specified by `dataset_name'.

    code recycled from George Papamakarios (https://github.com/gpapamak/maf)

    Args:
        dataset_name: string, see below for options
        frac: float, fraction of training dataset to use (useful for debugging)
        data_args: arguments to data constructor (see below for example)

    return: train_dp, val_dp, test_dp
    """

    assert isinstance(dataset_name, str), 'Name must be a string'

    if dataset_name == '1d_gauss':
        data_source = data_handlers.OneDimMoG(**data_args)

    elif dataset_name == '2d_mog':
        data_source = data_handlers.MoG(**data_args)

    elif dataset_name == '2d_spiral':
        data_source = data_handlers.Spiral(**data_args)

    elif dataset_name == 'mnist':
        data_source = data_handlers.MNIST(**data_args)

    elif dataset_name == "multiomniglot":
        data_source = data_handlers.MultiOmniglot(**data_args)

    elif dataset_name == 'gaussians':
        data_source = data_handlers.GAUSSIANS(**data_args)

    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    modes = ["trn", "val", "tst"]
    dps = []
    for mode in modes:
        frac = 1.0 if mode == "tst" else frac  # use a fraction of the data (for fast testing)
        dps.append(DataProviderFromSource(
            data_source=data_source, batch_size=batch_size, use_labels=use_labels,
            mode=mode, frac=frac, seed=seed, shuffle_order=shuffle)
        )

    return dps  # triple of (trn_dp, val_dp, tst_dp)
