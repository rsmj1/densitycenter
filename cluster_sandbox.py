import numpy as np

from sklearn.metrics import normalized_mutual_info_score as nmi
from datetime import datetime
import time

from density_tree import make_tree
from n_density_tree import make_n_tree

from tree_plotting import plot_embedding
from cluster_tree import dc_clustering
from visualization import visualize, plot_tree, plot_tree_v2
from benchmark import create_dataset

#Algorithms
from hdbscan import HDBSCAN as HDBSCAN
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from DBSCAN import DBSCAN
from kcentroids import DCKCentroids
from HDBSCANnary import HDBSCAN as HDBSCANnary
from kcentroids_nary import DCKCentroids as DCKCentroids_nary
from kneed import KneeLocator



def runtime_save(points, labels, load_dataset):
    '''
    Function used to save the provided dataset during runtime. 
    Can be "plugged in" after any plot that might make you want to decide on whether to keep the dataset or not.
    Will prompt a commandline interaction.
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

def color_labels(points, color):
      '''
      Creates an array of the provided color that is as long as the number of points provided.
      The color choices are any that matplotlib recognizes.
      '''
      return [color for p in points]

def diff_labels(points):
      return np.arange(len(points))

def num_clusts(labels):
    num_clusters = len(np.unique(labels))
    if np.isin(-1, labels) and num_clusters != 1: #Should not count noise labels as a set of labels
                num_clusters -= 1
    return num_clusters

if __name__ == '__main__': 
    #################### RUN PARAMETERS HERE #######################
    k = 5
    min_pts = 3
    mcs = 2
    eps = 2


    #################### DATASET SETUP ####################
    #Toy datasets

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
    #                    [19,18]])

    #Interesting example with min_pts = 3, mcs = 2 - shows a potential bug for sklearn HDBSCAN.
    # points = np.array([[1,2],
    #                    [1,4],
    #                    [2,3],
    #                    [1,1],
    #                    [3.5,0.5], #5
    #                    [7.5,-3],
    #                    [40,40],
    #                    [41,41],
    #                    [40,41],
    #                    [40.5,40.5], #10
    #                    [30,30],
    #                    [15,18]])

    # points = np.array([[1,2],
    #                    [1,3],
    #                    [2,3],
    #                    [3,4], 
    #                    [7.5,-3], #5 
    #                    [21,21],
    #                    [20,21],
    #                    [20.5,20.5],
    #                    [19.5,19.5], 
    #                    [15,15], #10
    #                    [11,11]])

    
    #Real datasets
    num_points = 12
    dataset_type = "compound" 
    
    save_dataset = False
    save_name = "debugstability" #Name that the dataset and accompanying images are saved with
    
    load_dataset = False #If true will override the other params and just load from the load_name.
    load_name = "blobs12"
    ground_truth_labels = None

    points, ground_truth_labels = create_dataset(num_points=num_points, datatype=dataset_type, save=save_dataset, load=load_dataset, save_name=save_name, load_name=load_name, num_classes=k)
    display_labels = color_labels(points, "lightgreen")
    random_labels = diff_labels(points)
    if ground_truth_labels is None:
          ground_truth_labels = random_labels



    #################### INITIALIZE TREE ####################
    #These two methods create the binary and n-ary dc-tree. The dc-dist-matrix is the same for both.
    root, dc_dist_matrix = make_tree(points, None, min_points=min_pts)
    n_root, dc_dist_matrix = make_n_tree(points, None, min_points=min_pts)


    #################### CLUSTERING ALGORITHMS ####################

    #K-center
    pred_labels, kcenter_centers, epsilons = dc_clustering(root, num_points=len(ground_truth_labels), k=k, min_points=min_pts,with_noise=True)


    #Sklearn HDBSCAN*
    sklearn_hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples=min_pts)
    sklearn_hdbscan.fit(points)
    sklearn_hdb_labels = sklearn_hdbscan.labels_

    #HDBSCAN*
    hdbscan_nary = HDBSCANnary(min_pts=min_pts, min_cluster_size=mcs, allow_single_cluster=False)
    hdbscan_nary.fit(points)
    hdb_labels = hdbscan_nary.labels_

    k_hdbscan = num_clusts(hdb_labels) #If you want to use the k of hdbscan for kmedian / kmeans

    #K-median
    kmedian = DCKCentroids_nary(k=k, min_pts=min_pts, loss="kmedian", noise_mode="none")
    kmedian.fit(points)
    kmedian_labels = kmedian.labels_
    kmedian_centers = kmedian.center_indexes
    kmedian_hierarchy = kmedian.define_cluster_hierarchy_nary(points)

    #Elbow method on K-median
    costs = kmedian.cost_decreases
    costs = costs[1:] #Remove the first np.inf
    k_values = list(range(2, len(costs) + 2))  # Assuming k starts at 2 and increments by 1
    k_elbow = KneeLocator(k_values, costs, curve='convex', direction='decreasing')
    kmedian_elbow = DCKCentroids_nary(k=k_elbow, min_pts=min_pts, loss="kmedian", noise_mode="none")
    kmedian_elbow.fit(points)
    elbow_labels = kmedian_elbow.labels_

    #K-means
    kmeans = DCKCentroids_nary(k=k, min_pts=min_pts, loss="kmeans", noise_mode="none")
    kmeans.fit(points)
    kmeans_labels = kmeans.labels_
    kmeans_centers = kmeans.center_indexes
    kmeans_hierarchy = kmeans.define_cluster_hierarchy_nary(points)

    
    # #Euclidean K-median
    # kmedoids = KMedoids(n_clusters=k, random_state=0).fit(points)
    # labels = kmedoids.labels_

    # #Euclidean K-means
    # eu_kmeans = KMeans(n_clusters=k, n_init="auto")
    # eu_kmeans.fit(points)
    # eu_labels = kmeans.labels_



    ################################### RESULTS VISUALIZATION #####################################
    if points.shape[0] < 50:        
        #plot_tree(root, labels=ground_truth_labels) #This method is used for the binary dc-tree *legacy code*
        #plot_tree_v2(n_root, labels=ground_truth_labels)
        plot_tree_v2(n_root, labels=hdb_labels, node_size=300)
        visualize(points, ground_truth_labels)

    #Plot the final clustering of the datapoints in 2D euclidean space.
    plot_embedding(
        points,
        [kmeans_labels],
        ['Dataset with ' + str(len(ground_truth_labels)) + ' points and ' + str(k) + " clusters"],
        centers=None,
        dot_scale=2, #6 used for small dataset, 2 used for 400 points
        annotations=False
    )


    runtime_save(points, ground_truth_labels, load_dataset) #Enable this to have the option to save a dataset during runtime via the command line.




