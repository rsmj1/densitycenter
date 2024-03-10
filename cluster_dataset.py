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
from visualization import visualize
from benchmark import create_dataset

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




    #################### RUN PARAMETERS HERE #######################
    #create_dataset setup:
    num_points = 16
    dataset_type = "moon" 
    save_dataset = False
    load_dataset = False #If true will override the other params and just load from the filename.
    save_name = "testingmoon" #Shared for name of images, filename to save the dataset into
    load_name = "test"

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
    points, labels = create_dataset(num_points=num_points, type=dataset_type, save=save_dataset, load=load_dataset, save_name=save_name, load_name=load_name)
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

    

    #Default k-value is 4
    #Change it by calling this python script with command-line argument ".....py --k valueOfK"
    #Or just by changing the default value above...

    #Default min_pts value is 3 
    #Again, change in command-line or changing default value above.

    #K-means clustering
    kmeans = KMEANS(k=args.k)
    kmeans.plusplus_dc_kmeans(points=points, minPts=args.min_pts, max_iters=100)

    kmeans_labels = kmeans.labels
    centers = kmeans.centers




    '''
    HDBSCAN clustering:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    Has following relevant arguments: 
    - min_cluster_size: default=5
    - min_samples: defaults to min_cluster_size. This functions as minPts.  
    - metric: default = 'euclidean'

    '''
    hdbscan = HDBSCAN(min_cluster_size=2, min_samples = args.min_pts)
    hdbscan.fit(points)
    hdb_labels = hdbscan.labels_

    num_clusters = len(np.unique(hdb_labels))

    #Kmeans using same number of clusters as hdbscan finds
    kmeans_hk = KMEANS(k=num_clusters)
    kmeans_hk.plusplus_dc_kmeans(points=points, minPts=args.min_pts, max_iters=100)
    kmeans_labels_hk = kmeans_hk.labels

    k = str(args.k)
    hk = str(num_clusters)

    print("HDB K:", hk)
    print("HBB labels:", hdb_labels)

    ################################### RESULTS VISUALIZATION #####################################
    #Plot the complete graph from the dataset with the specified distance measure on all of the edges. Optionally show the distances in embedded space with MDS.
    visualize(points=points, cluster_labels=kmeans_labels, minPts=args.min_pts, distance="dc_dist", centers=centers, save=save_visualization, save_name=image_save_name)


    #Plot the dc-tree, optionally with the centers from the final kmeans clusters marked in red
    #plot_tree(root, kmeans_labels, kmeans.center_indexes, save=save_visualization, save_name=image_save_name)
    plot_tree(root, hdb_labels, None, save=save_visualization, save_name=image_save_name)


    #Plot the final clustering of the datapoints in 2D euclidean space.
    plot_points = points
    plot_embedding(
        plot_points,
        [labels   , pred_labels             , kmeans_labels , hdb_labels    , kmeans_labels_hk],
        ['truth'+k, 'k-Center on DC-dists'+k, 'K-means'+k   , 'HDBSCAN' + hk, 'K-means'+ hk],
        centers=centers,
        dot_scale=0.5
    )