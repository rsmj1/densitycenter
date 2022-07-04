import os
import time
from experiment_utils.metrics import classifier_accuracy, cluster_quality, cluster_distances
from experiment_utils.general_utils import get_ab, make_plot
from experiment_utils.get_data import get_dataset
from experiment_utils.get_algorithm import get_algorithm

import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['mnist', 'fashion_mnist', 'cifar', 'swiss_roll', 'coil', 'google_news'],
        default='coil',
        help='Which dataset to apply algorithm to'
    )
    parser.add_argument(
        '--dr-algorithm',
        choices=[
            'gidr_dun',
            'original_umap',
            'original_tsne',
            'pca',
            'kernel_pca',
            'rapids_umap',
            'rapids_tsne'
        ],
        default='gidr_dun',
        help='Which algorithm to use for performing dim reduction'
    )
    parser.add_argument(
        '--random-init',
        action='store_true',
        help='If true, perform random init. If false, do Lap. Eigenmaps'
    )
    parser.add_argument(
        '--tsne-symmetrization',
        action='store_true',
        help='When present, symmetrize using the tSNE method'
    )
    parser.add_argument(
        '--make-plots',
        action='store_true',
        help='When present, make plots regarding distance relationships. ' \
             'Requires a high downsample_stride to not run out of memory'
    )

    parser.add_argument(
        '--umap-metric',
        action='store_true',
        help='If true, subtract rho\'s to calculate the umap pseudo-distance metric'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='If true, runs optimization on the GPU. Requires that GPU setup has been performed.'
    )
    parser.add_argument(
        '--torch',
        action='store_true',
        help='If true, run optimization in pytorch rather than cython/numba/cuda'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=-1,
        help='If running CPU optimization in parallel, specifies the number of threads used'
    )

    parser.add_argument(
        '--optimize-method',
        choices=[
            'umap',
            'tsne',
            'gidr_dun', ],
        default='gidr_dun',
        help='Which embedding optimization algorithm to use'
    )
    parser.add_argument(
        '--numba',
        action='store_true',
        help="If present, run under-optimized numba versions of the algorithm. NOTE: not compatible with Barnes-hut TSNE"
    )
    parser.add_argument(
        '--normalized',
        action='store_true',
        help='If true, normalize high- and low-dimensional pairwise likelihood matrices'
    )
    parser.add_argument(
        '--sym-attraction',
        action='store_true',
        help='Whether to attract along both ends of a nearest neighbor edge'
    )
    parser.add_argument(
        '--frobenius',
        action='store_true',
        help='If true, calculate gradients with respect to frobenius norm rather than KL'
    )
    parser.add_argument(
        '--angular',
        action='store_true',
        help='When present, use cosine similarity metric on high-dimensional points'
    )
    parser.add_argument(
        '--tsne-scalars',
        action='store_true',
        help='true => a = b = 1; false => determine a, b'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=-1,
        help='Number of samples to use from the dataset'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15
    )
    parser.add_argument(
        '--neg-sample-rate',
        type=int,
        default=5,
        help='How many negative samples to use for each positive sample. Only applies for original UMAP'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=-1,
        help='Number of times to cycle through the dataset'
    )
    args = parser.parse_args()
    return args


def subsample_points(points, labels, num_classes, points_per_class, class_list=[]):
    if not class_list:
        all_classes = np.unique(labels)
        class_list = np.random.choice(all_classes, num_classes, replace=False)

    per_class_samples = [np.where(labels == sampled_class)[0] for sampled_class in class_list]
    min_per_class = min([len(s) for s in per_class_samples])
    per_class_samples = [s[:min_per_class] for s in per_class_samples]
    sample_indices = np.squeeze(np.stack([per_class_samples], axis=-1))
    total_points_per_class = int(sample_indices.shape[-1])
    if points_per_class < total_points_per_class:
        stride_rate = float(total_points_per_class) / points_per_class
        class_subsample_indices = np.arange(0, total_points_per_class, step=stride_rate).astype(np.int32)
        sample_indices = sample_indices[:, class_subsample_indices]

    sample_indices = np.reshape(sample_indices, -1)
    points = points[sample_indices]
    labels = labels[sample_indices]
    return points, labels


if __name__ == '__main__':
    args = get_args()

    print('Loading %s dataset...' % args.dataset)
    points, labels = get_dataset(args.dataset, args.num_points)
    points, labels = subsample_points(points, labels, 20, 72)
    a, b = get_ab(args.tsne_scalars)
    args_dict = vars(args)
    args_dict['a'] = a
    args_dict['b'] = b
    if args_dict['n_epochs'] < 0:
        args_dict['n_epochs'] = None
    # We amplify grads in case of normalization
    args_dict['amplify_grads'] = args_dict['normalized']
    dr = get_algorithm(args.dr_algorithm, args_dict)

    print('fitting...')
    start = time.time()
    dr_output = dr.fit_transform(points)
    if isinstance(dr_output, tuple):
        embedding, opt_time = dr_output
    else:
        embedding = dr_output
        opt_time = -1
    end = time.time()
    total_time = end - start
    print('Optimization took {:.3f} seconds'.format(opt_time))
    print('Total time took {:.3f} seconds'.format(total_time))

    save_dir = os.path.join('outputs', 'scratch')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'embedding.png')
    make_plot(embedding, labels, show_plot=True, save_path=save_path)
