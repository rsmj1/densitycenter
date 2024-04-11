import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.manifold import MDS
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx
from datetime import datetime
import time

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_dc_dist_matrix
from density_tree import make_tree
from tree_plotting import plot_embedding
from tree_plotting import plot_tree
from cluster_tree import dc_clustering
from point_gen import create_hierarchical_clusters
from visualization import visualize
from benchmark import create_dataset
from benchmark import normalize_cluster_ordering


#Algorithms
from sklearn.cluster import SpectralClustering
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans

from kmeans import DCKMeans
from kmedian import DCKMedian
from DBSCAN import DBSCAN
from HDBSCAN import HDBSCAN as newScan


if __name__ == '__main__': 
    #################### RUN PARAMETERS HERE #######################

    num_points = 50
    k = 4
    min_pts = 5
    mcs = 2

    plot_tree_bool = False
    n_neighbors = 15
    eps = 2
    dataset_type = "moon" 
    save_dataset = False
    load_dataset = False #If true will override the other params and just load from the filename.
    save_name = "debug" #Shared for name of images, filename to save the dataset into
    load_name = "debug"

    #visualization parameters - comment in or out the visualization tools in the section below
    save_visualization = False
    image_save_name = save_name

    '''
    Functions that are run:
    create_dataset: Creates the dataset
    make_tree: Creates the dc-tree for methods using the old dc-tree setup

    dc_clustering: makes k-center optimal algorithm clustering
    kmeans.naive_dc_kmeans: makes k-means lloyds algorithm
    hdbscan.fit: makes hdbscan in euclidean space which implicitly uses dc-dist in the algorithm
    kmeans_hk.naive_dc_kmeans: makes k-means with no of clusters hdbscan found

    plot_tree: Shows the dc-tree, optionally with centers if provided
    visualize: Shows the complete graph with chosen distance function
    plot_embedding: Shows the chosen embeddings
    '''

    #################### RUN PARAMETERS END  #######################



    #Create the dataset and old dc_tree setup for methods that need it as input
    points = np.array([[1,2],
                       [1,4],
                       [2,3],
                       [1,1],
                       [-5,15], #5
                       [11,13],
                       [13,11],
                       [10,8],
                       [14,13],
                       [16,17], #10
                       [18,19],
                       [19,18],
                       [0,4.2],
                       [0,4.1],
                       [0,4],
                       [0,4.21],
                       [0,4.18]
                       ]
                       )
    labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    # points, labels = create_dataset(num_points=num_points, type=dataset_type, save=save_dataset, load=load_dataset, save_name=save_name, load_name=load_name)

    # points = np.array([[1,2],
    #                    [2,1],
    #                    [10,11],
    #                    [11,10],
    #                    [20,21], #5
    #                    [21,20],
    #                    ])
    # labels = np.array([0,1,2,3,4,5])

    # points = np.array([[0,1],
    #                    [0,2],
    #                    [0,3],
    #                    [1,2],
    #                    [2,1], #5
    #                    [2,2],
    #                    [2,3],
    #                    [3,2],
    #                    [4,1],
    #                    [4,2], #10
    #                    [4,3],
    #                    [4,4]]) # This set makes an error in dc_clustering
    # labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

    root, dc_dists = make_tree(
    points,
    labels,
    min_points=min_pts,
    make_image=plot_tree_bool,
    n_neighbors=n_neighbors
    )

    
    #K-center
    pred_labels, kcenter_centers, epsilons = dc_clustering(root, num_points=len(labels), k=k, min_points=min_pts,with_noise=False)

    #print("Pred labels:", pred_labels)


    #K-means clustering
    kmeans = DCKMeans(k=k, min_pts=min_pts)
    kmeans.plusplus_dc_kmeans(points=points, minPts=min_pts, max_iters=100)

    kmeans_labels = kmeans.labels_
    centers = kmeans.centers


    #K-median clustering
    kmedian = DCKMedian(k=k, min_pts=min_pts)
    kmedian.fit(points)

    kmedian_labels = kmedian.labels_
    kmedian_centers = kmedian.center_indexes
    #print("kmedian labels:", kmedian_labels)

    '''
    HDBSCAN clustering:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    Has following relevant arguments: 
    - min_cluster_size: default=5
    - min_samples: defaults to min_cluster_size. This functions as minPts.  
    - metric: default = 'euclidean'

    '''

    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    dbscan.fit(points)
    dbscan_labels = dbscan.labels_


    hdbscan_new = newScan(min_pts=min_pts, min_cluster_size=mcs)
    hdbscan_new.fit(points)
    hdb_new_labels = hdbscan_new.labels_
    num_clusters_new = len(np.unique(hdb_new_labels))
    if np.isin(-1, hdb_new_labels) and num_clusters_new != 1: #Should not count noise labels as a set of labels
                num_clusters_new -= 1
    #visualize(points=points, cluster_labels=None, minPts=min_pts, distance="dc_dist", centers=centers, save=save_visualization, save_name=image_save_name)

    #plot_tree(root, kmedian_labels, kmedian_centers, save=save_visualization, save_name=image_save_name)
    #raise AssertionError("stop")

    hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples=min_pts)
    hdbscan.fit(points)
    hdb_labels = hdbscan.labels_

    num_clusters = len(np.unique(hdb_labels))
    if np.isin(-1, hdb_labels) and num_clusters != 1: #Should not count noise labels as a set of labels
                num_clusters -= 1
    #Kmeans using same number of clusters as hdbscan finds
    kmeans_hk = DCKMeans(k=num_clusters, min_pts=min_pts)
    kmeans_hk.plusplus_dc_kmeans(points=points, minPts=min_pts, max_iters=100)
    kmeans_labels_hk = kmeans_hk.labels_

    k = str(k)
    hk = str(num_clusters)
    hk_new = str(num_clusters_new)

    print("equal if zero:", np.count_nonzero(normalize_cluster_ordering(hdb_labels) -normalize_cluster_ordering(hdb_new_labels)) )
    print("New:", normalize_cluster_ordering(hdb_new_labels))
    print("Old:", normalize_cluster_ordering(hdb_labels))

    ################################### RESULTS VISUALIZATION #####################################
    #Plot the complete graph from the dataset with the specified distance measure on all of the edges. Optionally show the distances in embedded space with MDS.
    if points.shape[0] < 20:
        visualize(points=points, cluster_labels=hdb_labels, minPts=min_pts, distance="dc_dist", centers=centers, save=save_visualization, save_name=image_save_name)
        plot_tree(root, hdb_new_labels, None, save=save_visualization, save_name=image_save_name)
        plot_tree(root, hdb_labels, None, save=save_visualization, save_name=image_save_name)

        #Plot the dc-tree, optionally with the centers from the final kmeans clusters marked in red
        #plot_tree(root, kmeans_labels, kmeans.center_indexes, save=save_visualization, save_name=image_save_name)
        #plot_tree(root, hdb_labels, None, save=save_visualization, save_name=image_save_name)

    #Plot the final clustering of the datapoints in 2D euclidean space.
    plot_points = points
    plot_embedding(
        plot_points,
        [labels                            , pred_labels             , kmeans_labels , hdb_new_labels         , hdb_labels  , kmeans_labels_hk],
        ['truth'+str(len(np.unique(labels))), 'k-Center on DC-dists'+k, 'K-means'+k   , '(new)HDBSCAN' + hk_new, "HDBSCAN"+hk, 'K-means'+ hk],
        centers=centers,
        dot_scale=0.5
    )