from gidr_dun.gidr_dun_ import DensityDR
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

def get_algorithm(algorithm_str, params, verbose=True):
    print(algorithm_str)
    if 'gidr_dun' in algorithm_str:
        dr = DensityDR(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                random_state=98765,
                random_init=params['random_init'],
                pseudo_distance=params['umap_metric'],
                tsne_symmetrization=params['tsne_symmetrization'],
                optimize_method=params['optimize_method'],
                negative_sample_rate=params['neg_sample_rate'],
                normalized=int(params['normalized']),
                sym_attraction=int(params['sym_attraction']),
                frob=int(params['frobenius']),
                num_threads=params['num_threads'],
                amplify_grads=int(params['amplify_grads']),
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'original_umap':
        dr = UMAP(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                init='random' if params['random_init'] else 'spectral',
                negative_sample_rate=params['neg_sample_rate'],
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'original_tsne':
        dr = TSNE(random_state=98765, verbose=3)
    elif algorithm_str == 'pca':
        dr = PCA()
    else:
        raise ValueError("Unsupported algorithm")

    return dr
