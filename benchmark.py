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
from itertools import chain, combinations
import efficientdcdist.dctree as dcdist



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
            #This does not have ground truth labels, so just use sklearn kmeans over euclidean distance as "ground truth"
            points = create_hierarchical_clusters(n=num_points, unique_vals=True)
            euclid_kmeans = KMeans(n_clusters=k)
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
            np.savetxt("savefiles/datasets/"+save_name+'.csv', points, delimiter=',')
            np.savetxt("savefiles/datasets/"+save_name+'_labels.csv', labels, delimiter=',')
        
    return points, labels


def normalize_cluster_ordering(cluster_labels):
    '''
    Normalizes the clustering labels so that the first cluster label encountered is labelled 0, the next 1 and so on. 
    Preserves noise labelled as -1. Useful for clustering comparisons.
    '''
    print("cluster_labels.shape", cluster_labels.shape)
    n = cluster_labels.shape[0]
    print("n:", n)
    cluster_index = 0
    cluster_index_mapping = {}
    mapped_labels = set()
    norm_cluster_labels = np.empty(n)

    for i, label in enumerate(cluster_labels):
        if label not in mapped_labels:
            #Create mapping for new encountered cluster
            mapped_labels.add(label)
            cluster_index_mapping[label] = cluster_index
            cluster_index += 1
        
        if label == -1:
            #Preserve noise labellings
            norm_cluster_labels[i] = -1
        else:
            norm_cluster_labels[i] = cluster_index_mapping[label]

    return norm_cluster_labels
####################################### END OF METHOD SETUPS ##############################################


#################### RUN PARAMETERS HERE #######################
#create_dataset setup: #TODO Make into run class possibly
k = 4
min_pts = 3
num_points = 16
plot_tree_bool = False
n_neighbors = 15
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
    min_points=min_pts,
    make_image=plot_tree_bool,
    n_neighbors=n_neighbors
)

#K-center
pred_labels, kcenter_centers, epsilons = dc_clustering(
    root,
    num_points=len(labels),
    k=k,
    min_points=min_pts,
)



#Default k-value is 4
#Change it by calling this python script with command-line argument ".....py --k valueOfK"
#Or just by changing the default value above...

#Default min_pts value is 3 
#Again, change in command-line or changing default value above.

#K-means clustering
kmeans = KMEANS(k=k)
kmeans.plusplus_dc_kmeans(points=points, minPts=min_pts, max_iters=100)

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
hdbscan = HDBSCAN(min_cluster_size=2, min_samples = min_pts)
hdbscan.fit(points)
hdb_labels = hdbscan.labels_

num_clusters = len(np.unique(hdb_labels))

#Kmeans using same number of clusters as hdbscan finds
kmeans_hk = KMEANS(k=num_clusters)
kmeans_hk.plusplus_dc_kmeans(points=points, minPts=min_pts, max_iters=100)
kmeans_labels_hk = kmeans_hk.labels

k = str(k)
hk = str(num_clusters)

print("HDB K:", hk)
print("HBB labels:", hdb_labels)

################################### RESULTS VISUALIZATION #####################################
#Plot the complete graph from the dataset with the specified distance measure on all of the edges. Optionally show the distances in embedded space with MDS.
visualize(points=points, cluster_labels=kmeans_labels, minPts=min_pts, distance="dc_dist", centers=centers, save=save_visualization, save_name=image_save_name)


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



def brute_force_comparision(num_points, min_pts):
    '''
    Check if solution to HDBSCAN is equal to K-means given the K output by HDBSCAN. 
    This entails running HDBSCAN, getting its label output and the number of clusters k. Then K-means is run in a brute-force manner given that k and all possible ways to choose centers given this, and finding the choice that yields the optimal loss. Then this solution should give the same clusters as HDBSCAN.
    If no such optimal solution exists, it saves the instance.

    '''
    dc_tree = dcdist.DCTree(points, min_points=min_pts, n_jobs=1)

    dataset_type = "moon"
    #Generate the dataset
    points, _ = create_dataset(num_points=num_points, type=dataset_type, save=False, load=False)

    
    #Run HDBSCAN
    hdbscan = HDBSCAN(min_cluster_size=1, min_samples = min_pts)
    hdbscan.fit(points)
    hdb_labels = hdbscan.labels_

    k = len(np.unique(hdb_labels))

    indexes = np.arrange(num_points)
    kmeans = KMEANS(k=k)
    best_loss = np.inf
    best_centers = None

    #Make for loop here, checking all possible ways to choose k centers (their indexes) and their optimum.
    for cluster_combo in combinations(indexes, k):
        print("combo:", cluster_combo)

        curr_loss = kmeans.kmeans_loss(points, cluster_combo, dc_tree)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_centers = cluster_combo.copy()
    
    #Check equality between best kmeans clusters and HDBSCAN solution TODO: At some point add version that also checks how similar they are, not just if equal or not.
    kmeans_labels = kmeans.assign_points(points, best_centers, dc_tree)



    return



'''
TODO:
Additionally, we should have a code framework to make comparisons of all the algorithms against one another.
The relevant algorithms are:
kmeans
kmeans on the dc-dists
hungry kmeans on the dc-dists
dbscan
hdbscan
We then want to compare their results against ground truths and each others. This should be a quantitative and qualitative comparison, with means over runs and random seeds, etc.
Interesting because it will force us to have a good appreciation of where each method finds different solutions.

- TODO: Also we should bruteforce compare HDBSCAN to K-means to look for the optimal solutions

- TODO: Also currently Sklearn.HDBSCAN has min_cluster_size of at least 2... So cannot compare directly currently, need a way to set it to 1.
'''