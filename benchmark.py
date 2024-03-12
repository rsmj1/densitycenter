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
from kmeans import DCKMeans
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from point_gen import create_hierarchical_clusters
from visualization import visualize, print_numpy_code
from itertools import chain, combinations
import efficientdcdist.dctree as dcdist
import csv




def create_dataset(num_points, type, save=False, load=False, save_name=None, load_name=None, num_classes=6, k=4):
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
        points = np.loadtxt("savefiles/datasets/"+load_name+'.csv', delimiter=',')
        labels = np.loadtxt("savefiles/datasets/"+load_name+'_labels.csv', delimiter=',')
        print("Loaded file \"" + load_name + "\", with "+ str(points.shape[0]) + " points.")
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
    #print("cluster_labels before:", cluster_labels)
    n = cluster_labels.shape[0]
    cluster_index = 0
    cluster_index_mapping = {}
    mapped_labels = set()
    mapped_labels.add(-1)
    norm_cluster_labels = np.empty(n, dtype=np.int64)

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

    #print("cluster_labels after:", norm_cluster_labels)
    return norm_cluster_labels


def equate_noise(noise_labels, noisee_labels):
    '''
    Makes the noise points in noise labels into noise points in noisee labels
    '''
    if noise_labels.shape[0] != noisee_labels.shape[0]:
       raise AssertionError("The two sets of labels should have the same lengths...")

    for i, noise in enumerate(noise_labels):
        if noise == -1:
            noisee_labels[i] = -1
    return noisee_labels



def brute_force_comparision(num_points, min_pts, max_iters=100):
    '''
    Check if solution to HDBSCAN is equal to K-means given the K output by HDBSCAN. 
    This entails running HDBSCAN, getting its label output and the number of clusters k. Then K-means is run in a brute-force manner given that k and all possible ways to choose centers given this, and finding the choice that yields the optimal loss. Then this solution should give the same clusters as HDBSCAN.
    If no such optimal solution exists, it saves the instance.

    '''
    is_equal = True
    counter = 0
    bad_dataset = None

    while is_equal or counter >= max_iters:

        dataset_type = "moon"
        #Generate the dataset
        points, _ = create_dataset(num_points=num_points, type=dataset_type, save=False, load=False)
        
        dc_tree = dcdist.DCTree(points, min_points=min_pts, n_jobs=1)

        
        #Run HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=2, min_samples = min_pts)
        hdbscan.fit(points)
        hdb_labels = hdbscan.labels_

        k = len(np.unique(hdb_labels))
        if np.isin(-1, hdb_labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1

        indexes = np.arange(num_points)
        kmeans = KMEANS(k=k)
        best_loss = np.inf
        best_centers = None

        #Make for loop here, checking all possible ways to choose k centers (their indexes) and their optimum.
        for c_c in combinations(indexes, k):
            cluster_combo = np.array(c_c)
            #print("combo:", cluster_combo)

            curr_loss = kmeans.kmeans_loss(points, cluster_combo, dc_tree)
            if curr_loss < best_loss:
                print("new best loss:", curr_loss, "with centers:", cluster_combo)
                best_loss = curr_loss
                best_centers = cluster_combo.copy()
        
        #Check equality between best kmeans clusters and HDBSCAN solution TODO: At some point add version that also checks how similar they are, not just if equal or not.
        kmeans_labels = kmeans.assign_points(points, best_centers, dc_tree)
        kmeans_labels_norm = normalize_cluster_ordering(kmeans_labels)
        hdb_labels_norm = normalize_cluster_ordering(hdb_labels)
        #kmeans_labels_norm = equate_noise(hdb_labels_norm, kmeans_labels_norm)
        print("kmeans_labels_norm:", kmeans_labels_norm)
        print("hdb_labels_norm:   ", hdb_labels_norm)
        #TODO Extract and understand issue instances.
        if not np.array_equal(kmeans_labels_norm, hdb_labels_norm):
            is_equal = False
            bad_dataset = points
        counter += 1
    
    if not is_equal:
        print("They were not equal in an instance with k =", k)
        print_numpy_code(points)
    else:
        print("They were equal across all "+str(counter)+" instances!")

    return

def benchmark(dataset_types, num_points, num_runs, runtypes, k, min_pts, eps, visualize_results=False, save_results=False, save_name="test", plot_embeddings = False):
    '''
    Runs a set of algorithms "runtypes" on a set of datasets "dataset_types" with "num_runs" iterations on each dataset. 
    Within an iteration i of num_runs, will use the k (number of clusters) output from DBSCAN or HDBSCAN on algorithms that come after it.
    Makes a comparison grid given a specified metric (TODO) between the set of labels averaging across the num_runs runs of all algorithms on a dataset. 
    Therefore, the result is  (len(runtypes)+1) x (len(runtypes)+1) x len(dataset_types). 
    Can output it to a CSV file, where the k, min_pts and eps values used for the run are appended to the algorithm name in the headers.

    Parameters
    ----------

    dataset_types : List: String 


    num_points : Int

    num_runs : Int

    runtypes : List: String

    metrics : List: String, default=["nmi"]

    k : Int

    min_pts : Int 

    eps : Float

    visualize_results : Boolean, default=False

    save_results : Boolean, default=False

    save_name : String, default="test"

    plot_embeddings : Boolean, default=False
    '''
    num_runtypes = len(runtypes)
    num_datasets = len(dataset_types)

    datasets = np.zeros((num_datasets*num_runs, num_points)) #TODO: Currently does not support variable length datasets. Stores the actual datasets.

    #If we want extra metrics should multiply depth by number of metrics. We want the layers for the same dataset across the metrics on top of each other.
    benchmark_results = np.zeros((num_runtypes+1, num_runtypes+1, num_datasets)) # Make square matrix with each layer being a separate dataset it has been run on. +1 for ground truth
    headers = np.empty((num_runtypes+1, num_datasets), dtype=str)
    #TODO: Put the metric information and number of runs averaged across in the headers as well.
    #TODO: add possibility to use plot_embedding that will pop up each time a set of runs has finished to visually compare clusterings.

    for d, dataset_type in enumerate(dataset_types):
        comparison_matrix = np.zeros((num_runtypes, num_runtypes, num_runs))

        for i in range(num_runs):
            curr_k = k #Reset k between each run
            points, ground_truths = create_dataset(num_points, dataset_type) #Move this outside num_runs if should be same across runs
            datasets[num_datasets*d + i,:] = points #Save the points generated for visualization later
            n = points.shape[0]

            headers[num_runtypes, d] = "Ground Truth k"+str(np.unique(ground_truths)) #Put ground truth as last element in square

            curr_labels = np.zeros((num_runtypes+1, n))
            curr_labels[num_runtypes] = ground_truths

            after_hdb = False
            for r, runtype in enumerate(runtypes):
                if runtype == "HDBSCAN" or runtype == "DBSCAN":
                    after_hdb = True #If algo run after HDB the K could be variable and should not be part of the header (if multiple runs)
                print("Running", runtype, "on dataset \"", dataset_type + "\" with", num_points, "points")
                labels, curr_k, used_min_pts, used_eps = benchmark_single(points, runtype, curr_k, min_pts, eps)
                #Add the output labels to the collection of labels for this current dataset
                curr_labels[r] = labels
                if i == 0 and num_runs == 1:
                    #Only 1 run, can always just put the k
                    headers[r, d] = runtype+"_"+curr_k+"_"+used_min_pts+"_"+used_eps
                elif i == 0:
                    #Multiple runs, only put k if before DB or HDB as they alter the K
                    if not after_hdb:
                        headers[r, d] = runtype+"_"+curr_k+"_"+used_min_pts+"_"+used_eps
                    else:
                        headers[r, d] = runtype+"_"+"dbK"+"_"+used_min_pts+"_"+used_eps

            
            #Do comparison between the different algorithms TODO: Should run this for each metric
            comparison_matrix[:,:,i] = metric_matrix(curr_labels)
            
        averaged_comparisons = np.mean(comparison_matrix, axis=2)
        benchmark_results[:,:,d] = averaged_comparisons #Insert the averaged values into the benchmark results



    if save_results:
        results_to_csv(benchmark_results, dataset_types, runtypes, save_name)
    
    display_results(benchmark_results, headers, dataset_types, datasets) #Should get as input the actual datasets to display the clusterings

    print_results(benchmark_results, dataset_types, runtypes)
    return

def results_to_csv(results, headers, datasets, file_name):
    with open("savefiles/benchmarks/"+file_name+".csv", "w", newline='') as res_file:
        writer = csv.writer(res_file)

        for i in range(results.shape[2]): #For each layer of the results
            writer.writerow([datasets[i]]+list(headers[:,i])) #Write the column header of the current layer for the current dataset

            for r, row in enumerate(results[:,:,i]): #For each of the rows in the current 2D layer
                writer.writerow([headers[r,i]] + list(row))
            writer.writerow([""]) #Empty row to separate results

        #reshaped_results = results.transpose((0,2,1)).reshape(results.shape[0], -1) #Take all layers and put beside each other from left to right, leftmost being topmost

    return


def display_results(results, headers, dataset_types, datasets):
    '''
    Displays the comparative cluster metrics in heatmap grids
    '''
    return

def print_results(results, row_headers, col_headers):
    '''
    Prints the benchmark data to the console for a quick glance.
    '''


    return

def metric_matrix(label_results):
    '''
    Creates a matrix of size num_labellings x num_labellings computing the NMI between each. 
    '''
    n = label_results.shape[0]
    comparison_matrix = np.zeros((n,n))
    for i, labels1 in enumerate(label_results):
        for j, labels2 in enumerate(label_results):
            comparison_matrix[i,j] = nmi(labels1, labels2)
            #Currently should probably just be NMI: It is symmetric. 
    return comparison_matrix


def benchmark_single(points, runtype, k, min_pts, eps):
    '''
    Runs the provided runtype on points and returns the k used in the algorithm and the resulting labels.
    '''
    labels = None
    used_min_pts = -1
    used_eps = -1
    if runtype == "HDBSCAN":
        hdbscan = HDBSCAN(min_cluster_size=2, min_samples = min_pts)
        hdbscan.fit(points)
        labels = hdbscan.labels_
        k = len(np.unique(labels))
        if np.isin(-1, labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1
        used_min_pts = min_pts
    elif runtype == "DBSCAN":
        dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        dbscan.fit(points)
        labels = dbscan.labels_
        k = len(np.unique(labels))
        if np.isin(-1, labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1
        used_min_pts = min_pts
        used_eps = eps
    elif runtype == "KMEANS":
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(points)
        labels = kmeans.labels_

    elif runtype == "DCKMeans":
        dckmeans = DCKMeans(k=k, min_pts=min_pts)
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts

    elif runtype == "HUNGRYKMEANS":
        print("TODO")

    return labels, k, used_min_pts, used_eps


if __name__ == "__main__":
    brute_force_comparision(num_points=10, min_pts=3)



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