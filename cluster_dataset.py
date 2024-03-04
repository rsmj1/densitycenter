import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.manifold import MDS
from sklearn.cluster import SpectralClustering
from DBSCAN import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx
from datetime import datetime
#import hdbscan

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_dc_dist_matrix
from density_tree import make_tree
from tree_plotting import plot_embedding
from tree_plotting import plot_tree
from cluster_tree import dc_clustering
#from GDR import GradientDR
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering

#My addons
from kmeans import KMEANS
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from point_gen import create_hierarchical_clusters

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--min-pts',
        type=int,
        default=3,
        help='Min points parameter to use for density-connectedness'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        help='Number of clusters for density-connected k-means'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15,
        help='Dummy variable for compatibility with UMAP/tSNE distance calculation'
    )
    parser.add_argument(
        '--plot-tree',
        action='store_true',
        help='If present, will make a plot of the tree'
    )
    args = parser.parse_args()

    #CHANGE NUMBER OF SAMPLES HERE
    samples = 16



    def create_dataset(num_points, type, save=False, load=False, save_name=None, load_name=None, num_classes=6):
        """
        Parameters
        ----------
        num_points : int
            The number of points to be created.
        
        type : String
            The dataset type. Options: moon, gauss, circle, synth, coil, mnist
        
        save : Bool, default=False
            If true, saves the generated dataset and its ground truth labels in .csv files under the provided save_name.
        
        load : Bool, default=False
            If true, loads dataset from .csv files with accompanied ground truth labels.

        save_name : String, default=None

        load_name : String, default=None

        num_classes: Int, default=6
            This is used if dataset is one of the following: synth, coil, mnist
        
        """

        points, labels = None, None
        if load:
            if load_name is None:
                raise ValueError("Load name should not be none when attempting to load a file..")
            points = np.loadtxt(load_name+'.csv', delimiter=',')
            labels = np.loadtxt(load_name+'_labels.csv', delimiter=',')
        else: 
            if type == "moon":
                points, labels = make_moons(n_samples=num_points, noise=0.1)
            elif type == "gauss":
                points = create_hierarchical_clusters(n=samples, unique_vals=True)
                euclid_kmeans = KMeans(n_clusters=args.k)
                euclid_kmeans.fit(points)
                labels = euclid_kmeans.labels_
            elif type == "circle":
                points, labels = make_circles(n_samples=num_points, noise=0.01, radii=[0.5, 1.0], thicknesses=[0.1, 0.1])
            elif type == "synth":
                points, labels = get_dataset('synth', num_classes=num_classes, points_per_class=(num_points//num_classes))
            elif type == "coil":
                points, labels = get_dataset('coil', class_list=np.arange(1, num_classes), points_per_class=(num_points//num_classes))
            elif type == "mnist":
                points, labels = get_dataset('mnist', num_classes=num_classes, points_per_class=(num_points//num_classes))
            
            if save:
                if save_name is None:
                    save_name = str(datetime.now())
                np.savetxt(save_name+'.csv', points, delimiter=',')
                np.savetxt(save_name+'_labels.csv', labels, delimiter=',')
          
        return points, labels


    
    
    #Point generation with hierarchical clusters, comment in or out these four lines
    #This does not have ground truth labels, so just use sklearn kmeans over euclidean distance as "ground truth"
    
    #End of point generation with hierarchical clusters
    points, labels = create_dataset(num_points=10, type="moon", save=False, load=False, save_name="test", load_name="test")
    root, dc_dists = make_tree(
        points,
        labels,
        min_points=args.min_pts,
        make_image=args.plot_tree,
        n_neighbors=args.n_neighbors
    )

    #K-center
    pred_labels, kcenter_centers, epsilons = dc_clustering(
        root,
        num_points=len(labels),
        k=args.k,
        min_points=args.min_pts,
    )

    # Change the eps by a tiny amount so that that distance is included in the DBSCAN cuts
    #eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8

    # DBSCAN*
    # dbscan_orig = DBSCAN(eps=eps, min_pts=args.min_pts, cluster_type='corepoints')
    # dbscan_orig.fit(points)

    # dbscan_core_pt_inds = np.where(dbscan_orig.labels_ > -1)
    # dc_core_pt_inds = np.where(np.logical_and(pred_labels > -1, dbscan_orig.labels_ > -1))

    # Ultrametric Spectral Clustering
    # no_lambdas = get_lambdas(root, eps)
    # dsnenns = get_dc_dist_matrix(points, args.n_neighbors, min_points=args.min_pts)
    # sim = get_sim_mx(dsnenns)
    # sc_, sc_labels = run_spectral_clustering(
    #     root,
    #     sim,
    #     dc_dists,
    #     eps=eps,
    #     it=no_lambdas,
    #     min_pts=args.min_pts,
    #     n_clusters=args.k,
    #     type_="it"
    # )

    ################################ MY CODE ####################################
    #Default k-value is 4
    #Change it by calling this python script with command-line argument ".....py --k valueOfK"
    #Or just by changing the default value above...

    #Default min_pts value is 3 
    #Again, change in command-line or changing default value above.

    #K-means clustering
    kmeans = KMEANS(k=args.k)
    kmeans.naive_dc_kmeans(points=points, minPts=args.min_pts, max_iters=100)
    #kmeans.plusplus_dc_kmeans(points=points, minPts=args.min_pts, max_iters=100)

    kmeans_labels = kmeans.labels
    centers = kmeans.centers





    #HDBSCAN clustering
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    Has following relevant arguments: 
    - min_cluster_size:
    - min_samples: defaults to min_cluster_size. This functions as minPts.  
    - metric: default = 'euclidean'

    '''
    hdbscan = HDBSCAN(min_samples = args.min_pts)
    hdbscan.fit(points)
    hdb_labels = hdbscan.labels_

    num_clusters = len(np.unique(hdb_labels))
    print("Number of clusters:", num_clusters)

    #Kmeans using same number of clusters as hdbscan finds
    kmeans2 = KMEANS(k=num_clusters)
    kmeans2.naive_dc_kmeans(points=points, minPts=args.min_pts, max_iters=100)
    kmeans_labels2 = kmeans2.labels


    plot_tree(root, kmeans_labels, kmeans.center_indexes)


    plot_points = points
    plot_embedding(
        plot_points,
        [labels, pred_labels, kmeans_labels, hdb_labels, kmeans_labels2],
        ['truth'+str(args.k), 'k-Center on DC-dists'+str(args.k), 'K-means'+str(args.k), 'HDBSCAN' + str(num_clusters), 'K-means'+ str(num_clusters)],
        centers=centers,
        dot_scale=0.5
    )



    ################################ MY CODE END ####################################




    # print('Epsilon values per clusters', epsilons)
    # print('NMI spectral vs. k-center:', nmi(sc_labels, pred_labels))
    # print('NMI spectral vs. DBSCAN*:', nmi(sc_labels, dbscan_orig.labels_))
    # print('NMI DBSCAN* vs. k-center:', nmi(dbscan_orig.labels_, pred_labels))

    # plot_points = points
    # plot_embedding(
    #     plot_points,
    #     [labels, pred_labels, dbscan_orig.labels_, sc_labels],
    #     ['truth', 'k-Center on DC-dists', 'DBSCAN*', 'Ultrametric Spectral Clustering'],
    #     centers=centers
    # )
