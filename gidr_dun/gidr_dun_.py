# Originally written by Leland McInnes <leland.mcinnes@gmail.com>
# Modified by Andrew Draganov <draganovandrew@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import locale
from warnings import warn
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba

from . import distances as dist
from . import utils
from . import spectral
from . import graph_weights

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float32)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def standardize_neighbors(graph):
    """
    FIXME
    """
    data = graph.data
    head = graph.row
    tail = graph.col
    min_neighbors = np.min(np.unique(tail, return_counts=True)[1])
    new_head, new_tail, new_data = [], [], []
    current_point = 0
    counter = 0
    for edge in range(head.shape[0]):
        if tail[edge] > current_point:
            current_point = tail[edge]
            counter = 0
        if counter < min_neighbors:
            new_head.append(head[edge])
            new_tail.append(tail[edge])
            new_data.append(data[edge])
            counter += 1
    return scipy.sparse.coo_matrix(
        (new_data, (new_head, new_tail)),
        shape=graph.shape
    )

def simplicial_set_embedding(
        optimize_method,
        normalized,
        euclidean,
        sym_attraction,
        frob,
        numba,
        torch,
        gpu,
        num_threads,
        amplify_grads,
        data,
        graph,
        n_components,
        initial_lr,
        a,
        b,
        negative_sample_rate,
        n_epochs,
        random_init,
        random_state,
        metric,
        output_metric=dist.named_distances_with_gradients["euclidean"],
        euclidean_output=True,
        parallel=False,
        verbose=False,
):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    print(5, "{:04f}".format(time.time()))
    start = time.time()
    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    if n_epochs is None:
        n_epochs = default_epochs

    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    print(6, "{:04f}".format(time.time()))
    if random_init:
        embedding = random_state.multivariate_normal(
            mean=np.zeros(n_components), cov=np.eye(n_components), size=(graph.shape[0])
        ).astype(np.float32)
    else:
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral.spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )

    print(7, "{:04f}".format(time.time()))
    # ANDREW - head and tail here represent the indices of nodes that have edges in high-dim
    #        - So for each edge e_{ij}, head is low-dim embedding of point i and tail
    #          is low-dim embedding of point j
    head = graph.row
    tail = graph.col
    neighbor_counts = np.unique(tail, return_counts=True)[1]

    # ANDREW - get number of epochs that we will optimize this EDGE for
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    # ANDREW - renormalize initial embedding to be in range [0, 10]
    embedding = (
            10.0
            * (embedding - np.min(embedding, 0))
            / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    if verbose:
        print('optimizing layout...')
    # FIXME FIXME -- need to make sure that all numpy matrices are in
    #   'c' format!
    print(8, "{:04f}".format(time.time()))
    embedding, opt_time = _optimize_layout_euclidean(
        optimize_method,
        normalized,
        euclidean,
        sym_attraction,
        frob,
        numba,
        torch,
        gpu,
        num_threads,
        amplify_grads,
        embedding,
        embedding,
        head,
        tail,
        graph.data,
        neighbor_counts,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        initial_lr,
        negative_sample_rate,
        verbose
    )

    print(9, "{:04f}".format(time.time()))
    return embedding, opt_time

def _optimize_layout_euclidean(
        optimize_method,
        normalized,
        euclidean,
        sym_attraction,
        frob,
        numba,
        torch,
        gpu,
        num_threads,
        amplify_grads,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        neighbor_counts,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        initial_lr,
        negative_sample_rate,
        parallel=False,
        verbose=True,
):
    weights = weights.astype(np.float32)
    args = {
        'optimize_method': optimize_method,
        'normalized': normalized,
        'angular': not euclidean,
        'sym_attraction': int(sym_attraction),
        'frob': int(frob),
        'num_threads': num_threads,
        'amplify_grads': int(amplify_grads),
        'head_embedding': head_embedding,
        'tail_embedding': tail_embedding,
        'head': head,
        'tail': tail,
        'weights': weights,
        'neighbor_counts': neighbor_counts,
        'n_epochs': n_epochs,
        'n_vertices': n_vertices,
        'epochs_per_sample': epochs_per_sample,
        'a': a,
        'b': b,
        'initial_lr': initial_lr,
        'negative_sample_rate': negative_sample_rate,
        'rng_state': rng_state,
        'verbose': int(verbose)
    }
    start = time.time()
    if gpu:
        if optimize_method != 'gidr_dun':
            raise ValueError('GPU optimization can only be performed in the gidr_dun setting')
        from optimize_gpu import gpu_opt_wrapper as optimizer
    elif torch:
        if optimize_method != 'gidr_dun':
            raise ValueError('PyTorch optimization can only be performed in the gidr_dun setting')
        from .pytorch_optimize import torch_optimize_layout as optimizer
    elif numba:
        if optimize_method == 'umap':
            from .numba_optimizers.umap import optimize_layout_euclidean as optimizer
        elif optimize_method == 'gidr_dun':
            from .numba_optimizers.gidr_dun import gidr_dun_numba_wrapper as optimizer
        else:
            raise ValueError('Numba optimization only works for umap and gidr_dun')
    else:
        if optimize_method == 'umap':
            from umap_opt import umap_opt_wrapper as optimizer
        elif optimize_method == 'tsne':
            from tsne_opt import tsne_opt_wrapper as optimizer
        elif optimize_method == 'gidr_dun':
            from gidr_dun_opt import gidr_dun_opt_wrapper as optimizer
        else:
            raise ValueError("Optimization method is unsupported at the current time")
    embedding = optimizer(**args)
    end = time.time()
    opt_time = end - start
    # FIXME -- make into logger output
    if verbose:
        print('Optimization took {:.3f} seconds'.format(opt_time))
    return embedding, opt_time

@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).

    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample

    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result

@numba.njit()
def init_update(current_init, n_original_samples, indices):
    for i in range(n_original_samples, indices.shape[0]):
        n = 0
        for j in range(indices.shape[1]):
            for d in range(current_init.shape[1]):
                if indices[i, j] < n_original_samples:
                    n += 1
                    current_init[i, d] += current_init[indices[i, j], d]
        for d in range(current_init.shape[1]):
            current_init[i, d] /= n

    return

def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

class UniformUmap(BaseEstimator):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean

    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    random_init: boolean (optional, default false)
        How to initialize the low dimensional embedding

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    low_memory: bool (optional, default True)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that UMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.

    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.

    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.

    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.

    verbose: bool (optional, default False)
        Controls verbosity of logging.

    unique: bool (optional, default False)
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
        For to map from internal structures back to your data use the variable
        _unique_inverse_.

    """

    def __init__(
            self,
            n_neighbors=15,
            n_components=2,
            # FIXME - we don't actually use this
            metric="euclidean",
            output_metric="euclidean",
            n_epochs=None,
            learning_rate=1.0,
            random_init=False,
            pseudo_distance=True,
            tsne_symmetrization=False,
            optimize_method='umap_sampling',
            normalized=0,
            euclidean=True,
            sym_attraction=True,
            frob=False,
            numba=False,
            torch=False,
            gpu=False,
            amplify_grads=False,
            min_dist=0.1,
            spread=1.0,
            low_memory=True,
            num_threads=-1,
            local_connectivity=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            random_state=None,
            target_n_neighbors=-1,
            target_metric="categorical",
            target_weight=0.5,
            transform_seed=42,
            force_approximation_algorithm=False,
            verbose=False,
            unique=False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.n_epochs = n_epochs
        self.random_init = random_init
        self.n_components = n_components
        self.learning_rate = learning_rate

        # ANDREW - options for flipping between tSNE and UMAP
        self.tsne_symmetrization = tsne_symmetrization
        self.pseudo_distance = pseudo_distance
        self.optimize_method = optimize_method
        self.normalized = normalized
        self.euclidean = euclidean
        self.sym_attraction = sym_attraction
        self.frob = frob
        self.numba = numba
        self.torch = torch
        self.gpu = gpu
        self.euclidean = euclidean
        self.amplify_grads = amplify_grads

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.num_threads = num_threads

        self.a = a
        self.b = b

    def _validate_parameters(self):
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.random_init, bool):
            raise ValueError("init must be a bool")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_lr < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
                self.n_epochs < 0 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a nonnegative integer")
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        # set input distance metric & inverse_transform distance metric
        if self.metric == "precomputed":
            if self.unique:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn(
                "using precomputed metric; inverse_transform will be unavailable"
            )
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric in dist.named_distances:
            # ANDREW - Euclidean metric leads us into this branch of the if statements
            self._input_distance_func = dist.named_distances[self.metric]
            try:
                self._inverse_distance_func = dist.named_distances_with_gradients[self.metric]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")

        # set output distance metric
        if callable(self.output_metric):
            self._output_distance_func = self.output_metric
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in dist.named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )

        if self.num_threads < -1 or self.num_threads == 0:
            raise ValueError("num_threads must be a postive integer, or -1 (for all cores)")

    def fit(self, X):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        self._initial_lr = self.learning_rate
        self._validate_parameters()
        if self.verbose:
            print(str(self))

        if self.numba:
            self._original_n_threads = numba.get_num_threads()
            if self.num_threads > 0 and self.num_threads is not None:
                numba.set_num_threads(self.num_threads)
            else:
                self.num_threads = self._original_n_threads

        index = list(range(X.shape[0]))
        inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self, 0

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Constructing nearest neighbor graph...")

        start = time.time()

        # Handle small cases efficiently by computing all distances
        self._small_data = False
        print(0, "{:04f}".format(time.time()))
        if self.gpu:
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            import cudf
            knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors)
            cu_X = cudf.DataFrame(X[index])
            knn_cuml.fit(cu_X)
            dists, inds = knn_cuml.kneighbors(X)
            self._knn_dists = np.reshape(dists, [X.shape[0], self._n_neighbors])
            self._knn_indices = np.reshape(inds, [X.shape[0], self._n_neighbors])
        else:
            self._knn_indices, self._knn_dists, self._knn_search_index = graph_weights.nearest_neighbors(
                X[index],
                self._n_neighbors,
                self.metric,
                self.euclidean,
                random_state,
                self.low_memory,
                use_pynndescent=True,
                num_threads=self.num_threads,
                verbose=True,
            )

        (
            self.graph_,
            self._sigmas,
            self._rhos,
            self.graph_dists_,
        ) = graph_weights.fuzzy_simplicial_set(
            X[index],
            self.n_neighbors,
            random_state,
            self.metric,
            self._knn_indices,
            self._knn_dists,
            self.local_connectivity,
            self.verbose,
            pseudo_distance=self.pseudo_distance,
            euclidean=self.euclidean,
            tsne_symmetrization=self.tsne_symmetrization,
            gpu=self.gpu,
        )
        # Report the number of vertices with degree 0 in our our umap.graph_
        # This ensures that they were properly disconnected.
        vertices_disconnected = np.sum(
            np.array(self.graph_.sum(axis=1)).flatten() == 0
        )

        end = time.time()
        if self.verbose:
            print('Calculating high dim similarities took {:.3f} seconds'.format(end - start))

        if self.verbose:
            print(utils.ts(), "Construct embedding")

        self.embedding_, opt_time = self._fit_embed_data(
            self._raw_data[index],
            self.n_epochs,
            random_state,
        )
        # Assign any points that are fully disconnected from our manifold(s) to have embedding
        # coordinates of np.nan.  These will be filtered by our plotting functions automatically.
        # They also prevent users from being deceived a distance query to one of these points.
        # Might be worth moving this into simplicial_set_embedding or _fit_embed_data
        disconnected_vertices = np.array(self.graph_.sum(axis=1)).flatten() == 0
        if len(disconnected_vertices) > 0:
            self.embedding_[disconnected_vertices] = np.full(
                self.n_components, np.nan
            )

        self.embedding_ = self.embedding_[inverse]

        if self.verbose:
            print(utils.ts() + " Finished embedding")

        if self.numba:
            numba.set_num_threads(self._original_n_threads)
        self._input_hash = joblib.hash(self._raw_data)
        self.opt_time = opt_time

        return self

    def _fit_embed_data(self, X, n_epochs, random_state):
        """A method wrapper for simplicial_set_embedding that can be
        replaced by subclasses.
        """
        return simplicial_set_embedding(
            self.optimize_method,
            self.normalized,
            self.euclidean,
            self.sym_attraction,
            self.frob,
            self.numba,
            self.torch,
            self.gpu,
            self.num_threads,
            self.amplify_grads,
            X,
            self.graph_,
            self.n_components,
            self._initial_lr,
            self._a,
            self._b,
            self.negative_sample_rate,
            n_epochs,
            self.random_init,
            random_state,
            self._input_distance_func,
            self._output_distance_func,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
        )

    def fit_transform(self, X):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        _ = self.fit(X)
        return self.embedding_