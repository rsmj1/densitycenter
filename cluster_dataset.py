import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.datasets import load_iris

from sklearn.manifold import MDS
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx
from datetime import datetime
import time

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_dc_dist_matrix
from density_tree import make_tree
from n_density_tree import make_n_tree, prune_n_tree

from tree_plotting import plot_embedding
from cluster_tree import dc_clustering
from point_gen import create_hierarchical_clusters
from visualization import visualize, plot_tree
from benchmark import create_dataset
from benchmark import normalize_cluster_ordering
from cluster_tree import copy_tree, prune_tree

#Algorithms
from sklearn.cluster import SpectralClustering
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans

from kmeans import DCKMeans
from kmedian import DCKMedian
from DBSCAN import DBSCAN
from HDBSCAN import HDBSCAN as newScan
from kcentroids import DCKCentroids
from HDBSCANnary import HDBSCAN as HDBSCANnary
from kcentroids_nary import DCKCentroids as DCKCentroids_nary



def runtime_save(points, labels, load_dataset):
    '''
    Function used to save the provided dataset during runtime. 
    Can be "plugged in" after any plot that might make you want to decide on whether to keep the dataset or not.
    '''
    if not load_dataset:
        while True:
            command = input("Enter 'save' to save data points, or 'exit' to exit: ")
            if command.lower() == 'save':
                save_name = str(datetime.now())
                save_name = input("Enter filename to save dataset: ")
                np.savetxt("savefiles/datasets/"+save_name+'.csv', points, delimiter=',')
                np.savetxt("savefiles/datasets/"+save_name+'_labels.csv', labels, delimiter=',')
                print("Saved the dataset.")
                break
            else:
                print("Did not save the dataset.") 
                break



if __name__ == '__main__': 
    #################### RUN PARAMETERS HERE #######################

    num_points = 50
    k = 8
    min_pts = 3
    mcs = 2

    plot_tree_bool = False  
    n_neighbors = 15
    eps = 2
    dataset_type = "circle" 
    save_dataset = False
    load_dataset = True #If true will override the other params and just load from the filename.
    save_name = "debugstability" #Shared for name of images, filename to save the dataset into
    load_name = "meeting_example"

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
    #Exposing parameters for this example are min_pts = 3, mcs = 2
    #Showing how noise clusters are a thing to be avoided and how we have a weird exception for the root cluster.
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
                       ]
                       )
    labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

    #Interesting example with min_pts = 3, mcs = 2 - shows a potential bug for sklearn HDBSCAN.
    points = np.array([[1,2],
                       [1,4],
                       [2,3],
                       [1,1],
                       [3.5,0.5], #5
                       [7.5,-3],
                       [40,40],
                       [41,41],
                       [40,41],
                       [40.5,40.5], #10
                       [30,30],
                       [15,18],
                       ]
                       )
    labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

    points = np.array([[1,2],
                       [1,4],
                       [2,3],
                       [7.5,-3], #5
                       [41,41],
                       [40,41],
                       [40.5,40.5], 
                       [30,30], 
                       [15,18], #10
                       ]
                       )
    labels = np.array([0,1,2,3,4,5,6,7,8])

    # points = np.array([[1,2],
    #                    [1,4],
    #                    [2,3],
    #                    [1,1],
    #                    [-5,15], #5
    #                    [11,13],
    #                    [13,11],
    #                    [10,8],
    #                    [14,13],
    #                    [16,17], #10
    #                    [18,19],
    #                    [19,18],
    #                    [21,24],
    #                    [11,17],
    #                    [28,21], #15
    #                    [10,18],
    #                    [7,7]
    #                    ]
    #                    )
    # labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    
    # #Very simple example for visualizing example which also visualizes what happens when creating the n-ary dc-tree. 
    # points = np.array([[2,3],
    #                    [3,4],
    #                    [5,3],
    #                    [1,1],
    #                    [2,2]
    #                    ]
    #                    )
    # labels = np.array([0,1,2,3,4])


    dataset_type = "con_circles" 
    #points, labels = create_dataset(num_points=num_points, datatype=dataset_type, save=save_dataset, load=load_dataset, save_name=save_name, load_name=load_name)



    visualize(points=points, cluster_labels=labels, minPts=min_pts, distance="mut_reach", centers=None, save=save_visualization, save_name=image_save_name)
    


    
    t1 = time.time()
    root, dc_dists = make_tree(points, labels, min_points=min_pts)
    t2 = time.time()
    n_root,_ = make_n_tree(points, None, min_points=min_pts)
    t3 = time.time()
    #plot_tree(root, labels)

    print("binary dctree time:", t2-t1)

    print("nary dctree time  :", t3-t2)



    #K-center
    pred_labels, kcenter_centers, epsilons = dc_clustering(root, num_points=len(labels), k=k, min_points=min_pts,with_noise=True)



    kmedian_nary = DCKCentroids_nary(k=k, min_pts=min_pts, loss="kmedian", noise_mode="none")
    kmedian_nary.fit(points)

    kmedian_labels_nary = kmedian_nary.labels_
    kmedian_centers_nary = kmedian_nary.center_indexes
    new_hierarchy = kmedian_nary.define_cluster_hierarchy_nary(points)
    #plot_tree(new_hierarchy, is_binary=False)
    new_hierarchy = kmedian_nary.define_cluster_hierarchy_binary(points)
    #plot_tree(new_hierarchy, is_binary=True)
    #K-means clustering
    kmeans = DCKCentroids(k=k, min_pts=min_pts, loss="kmeans", noise_mode="none")
    kmeans.fit(points)

    kmeans_labels = kmeans.labels_
    centers = kmeans.centers
    

    #K-median clustering
    kmedian = DCKCentroids(k=k, min_pts=min_pts, loss="kmedian", noise_mode="full")
    kmedian.fit(points)

    kmedian_labels = kmedian.labels_
    kmedian_centers = kmedian.center_indexes


    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    dbscan.fit(points)
    dbscan_labels = dbscan.labels_

    #THIS CREATES ITS OWN TREE WITH ANOTHER STRUCTURE!!!
    hdbscan_new = newScan(min_pts=min_pts, min_cluster_size=mcs, allow_single_cluster=True, tree=root)
    hdbscan_new.fit(points)
    hdb_new_labels = hdbscan_new.labels_
    

    hdbscan_nary = HDBSCANnary(min_pts=min_pts, min_cluster_size=mcs, allow_single_cluster=False)
    hdbscan_nary.fit(points)
    hdbscan_nary_labels = hdbscan_nary.labels_

    num_clusters_new = len(np.unique(hdbscan_nary_labels))
    if np.isin(-1, hdbscan_nary_labels) and num_clusters_new != 1: #Should not count noise labels as a set of labels
                num_clusters_new -= 1

    hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples=min_pts)
    hdbscan.fit(points)
    hdb_labels = hdbscan.labels_

    num_clusters = len(np.unique(hdb_labels))
    if np.isin(-1, hdb_labels) and num_clusters != 1: #Should not count noise labels as a set of labels
                num_clusters -= 1
    
    
    #Kmeans using same number of clusters as hdbscan finds
    kmeans_hk = DCKCentroids(k=k, min_pts=min_pts, loss="kmeans", noise_mode="none")
    kmeans_hk.fit(points=points)
    kmeans_labels_hk = kmeans_hk.labels_

    k = str(k)
    hk = str(num_clusters)
    hk_new = str(num_clusters_new)
    


    ################################### RESULTS VISUALIZATION #####################################
    #Plot the complete graph from the dataset with the specified distance measure on all of the edges. Optionally show the distances in embedded space with MDS.
    if points.shape[0] < 100:
        #visualize(points=points, cluster_labels=hdb_labels, minPts=min_pts, distance="dc_dist", centers=centers, save=save_visualization, save_name=image_save_name)
        print("kmedian centers:", np.array(kmedian_nary.center_indexes)+1)
        plot_tree(n_root, labels=hdbscan_nary_labels, centers=None, is_binary=False)

        plot_tree(n_root, labels=hdbscan_nary_labels, centers = None, is_binary=False, extra_annotations=hdbscan_nary.extra_annotations)

        runtime_save(points, labels, load_dataset) #Enable this to have the option to save a dataset during runtime via the command line.


    #Plot the final clustering of the datapoints in 2D euclidean space.
    plot_points = points
    plot_embedding(
        plot_points,
        [hdbscan_nary_labels         , hdb_labels],
        ['HDBSCAN nary', "HDBSCAN"],
        centers=centers,
        dot_scale=1
    )

    plot_embedding(
        plot_points,
        [kmedian_labels, kmeans_labels],
        ['K-median' + k, "K-means" + k],
        centers=centers,
        dot_scale=1
    )

    plot_embedding(
        plot_points,
        [labels                            , pred_labels             , kmeans_labels , hdb_new_labels         , hdb_labels  , kmeans_labels_hk],
        ['truth'+str(len(np.unique(labels))), 'k-Center on DC-dists'+k, 'K-means'+k   , '(new)HDBSCAN' + hk_new, "HDBSCAN"+hk, 'K-means'+ hk],
        centers=centers,
        dot_scale=0.5
    )



