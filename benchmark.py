import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons, make_blobs, make_gaussian_quantiles, make_friedman1, make_classification, make_regression, make_s_curve
from sklearn.datasets import make_circles as make_circles_sklearn
from sklearn.manifold import MDS
from sklearn.cluster import SpectralClustering
from DBSCAN import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.decomposition import PCA
import networkx as nx
from datetime import datetime
#import hdbscan
import os, csv, glob

from experiment_utils.get_data import get_dataset, make_circles
from kneed import KneeLocator


from distance_metric import get_dc_dist_matrix
from density_tree import make_tree
from tree_plotting import plot_embedding
from visualization import  plot_tree
from cluster_tree import dc_clustering
#from GDR import GradientDR
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering
from sklearn_extra.cluster import KMedoids

#My addons
from kmeans import DCKMeans
from hdbscan import HDBSCAN as HDBSCAN
from sklearn.cluster import KMeans
from HDBSCAN_old import HDBSCAN as newScan
from kmedian import DCKMedian
from kcentroids import DCKCentroids
from cluster_tree import dc_clustering
from kcentroids_nary import DCKCentroids as DCKCentroidsNary
from HDBSCANnary import HDBSCAN as HDBSCANNary


from point_gen import create_hierarchical_clusters
from visualization import visualize, print_numpy_code
from itertools import chain, combinations
import efficientdcdist.dctree as dcdist
import csv
import warnings
warnings.filterwarnings('ignore')



def create_dataset(num_points, datatype, save=False, load=False, save_name=None, load_name=None, num_classes=6, k=4, num_features=2, noise=0.1):
    """
    Parameters
    ----------
    num_points : int
        The number of points to be created.
    
    type : String
        The dataset type. Options: 
         2d: moon, gauss, circle, con_circles, blobs, gauss_quantiles, classification, regression, compound, worms, aggregate
        >2d: swiss_rolls, s_curve, friedman, synth, coil, mnist
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
        ##### Synthetic datasets #####

        if datatype == "moons":
            #Creates two half-moons in a "yin-yang" shape
            points, labels = make_moons(n_samples=num_points, noise=noise)
        elif datatype == "gauss":
            #This does not have ground truth labels, so just use sklearn kmeans over euclidean distance as "ground truth"
            points = create_hierarchical_clusters(n=num_points, unique_vals=True)
            euclid_kmeans = KMeans(n_clusters=k)
            euclid_kmeans.fit(points)
            labels = euclid_kmeans.labels_
        elif datatype == "circle":
            #Generates circles - with given thicknesses and radii. This is a "homebrewed method"
            points, labels = make_circles(n_samples=num_points, noise=noise, radii=[0.5, 1.0], thicknesses=[0.1, 0.1])
        elif datatype == "con_circles":
            #Generates two circles within each other (concentric circles) - 2d circles
            points, labels = make_circles_sklearn(n_samples=num_points, noise=noise)
        elif datatype == "blobs":
            #The cluster_std is the density of each blob essentially.
            points, labels = make_blobs(n_samples=num_points, centers=10, n_features=num_features, random_state=1)
        elif datatype == "blobs2":
            cluster_numbers_blobs_unbalanced = [92, 157, 91, 133, 53, 70, 139, 74, 80, 111] #1000 in total
            points, labels = make_blobs(n_samples=cluster_numbers_blobs_unbalanced, centers=None, n_features=num_features, random_state=1)
        elif datatype == "gauss_quantiles":
            points, labels = make_gaussian_quantiles(n_samples=num_points, n_features=num_features, n_classes=num_classes, cov=2.0)
        elif datatype == "classification":
            #n_informative is relative to n_features the amount of the features that actually help in the classification
            if num_features == 2:
                num_classes = 2
            points, labels = make_classification(n_samples=num_points, n_features=num_features, n_informative=num_features//2, n_redundant=0, n_classes=num_classes, n_clusters_per_class=1)
        elif datatype == "regression":
            #A regression problem, similar thought process to classification above.
            points, labels = make_regression(n_samples=num_points, n_features=num_features, n_informative=num_features//2, n_redundant=0, n_classes=num_classes)

        # More than 2 features: Not for plotting. Synthetic 
        elif datatype == "swiss_rolls":
            #Literally creates a "roll" of the points in 3d - 3 features for each point.
            points, labels = make_swiss_roll(n_samples=num_points, noise=noise)
        elif datatype == "s_curve":
            #Outputs 3d points in an s-curve shape.
            points, labels = make_s_curve(n_samples=num_points, noise=noise)
        elif datatype == "friedman":
            #Has at least 5 features and generates points from random vectors with each entry in [0,1] transformed via: 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).
            #The rest of the features are not transformed.
            if num_features < 5:
                points, labels = make_friedman1(n_samples=num_points, n_features=5, noise=noise)
            else:
                points, labels = make_friedman1(n_samples=num_points, n_features=num_features, noise=noise)
        
        #Datasets from https://cs.joensuu.fi/sipu/datasets/ These are also 2d.
        elif datatype == "compound":
            points, labels = load_txt_datasets("compound")
        elif datatype == "worms": #No labels!!
            points, _ = load_txt_datasets("worms_2d")
            euclid_kmeans = KMeans(n_clusters=k)
            euclid_kmeans.fit(points)
            labels = euclid_kmeans.labels_
        elif datatype == "aggregate":
            points, labels = load_txt_datasets("Aggregation")
        elif datatype == "d31":
            points, labels = load_txt_datasets("d31")
        elif datatype == "overlap": #No labels!!
            points, labels = load_txt_datasets("overlap")
            points = np.unique(points, axis=0)
            labels = np.zeros(len(points))
        elif datatype == "jain":
            points, labels = load_txt_datasets("jain")
        elif datatype == "pathbased":
            points, labels = load_txt_datasets("pathbased")
        elif datatype == "spiral":
            points, labels = load_txt_datasets("spiral")

        elif datatype == "testing":
            points = np.array([[1,2], [1,4],[2,3],[1,1],[3.5,0.5],[7.5,-3],[40,40],[41,41],[40,41],[40.5,40.5],[30,30],[15,18],])
            labels = np.array([3,1,2,0,4,5,6,7,11,9,10,8])
        ##### Toy datasets #####
        elif datatype == "synth":
            points, labels = get_dataset('synth', num_classes=num_classes, points_per_class=(num_points//num_classes))
        elif datatype == "mnist":
            #Just loads the mnist dataset, ignoring the other parameters in this call.
            points, labels = get_dataset('mnist', num_classes=num_classes, points_per_class=(num_points//num_classes))


        ##### Benchmark datasets #####
        elif datatype == "coil": 
            #The coil100 dataset
            points, labels = get_dataset('coil', class_list=np.arange(1, num_classes), points_per_class=(num_points//num_classes))
        else:
            raise AssertionError("Dataset type \"" + datatype + "\" does not exist. Stopping...")
        
        if save:
            if save_name is None:
                save_name = str(datetime.now())
            np.savetxt("savefiles/datasets/"+save_name+'.csv', points, delimiter=',')
            np.savetxt("savefiles/datasets/"+save_name+'_labels.csv', labels, delimiter=',')

    return points, labels

def load_txt_datasets(dataset="compound"):
    '''
    Currently only works for 2d points with or without ground truth labels.
    '''
    points, labels = [],[]

    data = []
    path = os.path.join("data", "Synthetic", dataset+".txt")

    with open(path, "r") as data:
        for point in data:
            #print("line:", point)
            dims = point.strip().split()

            if len(dims) == 3:
                points.append(list(map(float, dims[:-1])))
                labels.append(int(dims[2]))
            else:
                points.append(list(map(float, dims)))
    if len(labels) != len(points):
        labels = np.zeros(len(points))
    return np.array(points), np.array(labels)

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
    Takes a set, noise labels, and overlays that noise in another set, noisee labels. 
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
        points, _ = create_dataset(num_points=num_points, datatype=dataset_type, save=False, load=False)
        
        dc_tree = dcdist.DCTree(points, min_points=min_pts, n_jobs=1)

        
        #Run HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=2, min_samples = min_pts)
        hdbscan.fit(points)
        hdb_labels = hdbscan.labels_

        k = len(np.unique(hdb_labels))
        if np.isin(-1, hdb_labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1

        indexes = np.arange(num_points)
        kmeans = DCKMeans(k=k)
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

def benchmark(dataset_types, num_points, num_runs, runtypes, k, min_pts, eps, mcs=None, num_features=None, metrics=["nmi"],visualize_results=False, save_results=False, save_name="test", plot_clusterings = False):
    '''
    Runs a set of algorithms "runtypes" on a set of datasets "dataset_types" with "num_runs" iterations on each dataset. 
    Within an iteration i of num_runs, will use the k (number of clusters) output from DBSCAN or HDBSCAN on algorithms that come after it.
    Makes a comparison grid for each specified metric between the set of labels averaging across the num_runs runs of all algorithms on a dataset. 
    Therefore, the result is  (len(runtypes)+1) x (len(runtypes)+1) x (len(dataset_types)*len(metrics)). Each layer of this 3-dimensional grid is a comparison matrix averaged over the number of runs for one of the chosen metrics.
    Can output it to a CSV file, where the k, min_pts and eps values used for the run are appended to the algorithm name in the headers in that order. -1 means not applicable for that algorithm.

    Parameters
    ----------

    dataset_types : List: String 
        The types of datasets to run the provided algorithms on.

    num_points : Int
        The number of points to instantiate each type of dataset with

    num_runs : Int
        The number of runs to perform on each dataset. The results over each run will be averaged. 
    
    runtypes : List: String
        The types of algorithms to run on the datasets. 
        Options: HDBSCAN, HDBSCAN_NEW, DBSCAN, KMEANS, DCKMEANS, DCKMEDIAN, DCKMEANS_MED, DCKMEDIAN_MED, DCKMEANS_FULL, DCKMEDIAN_FULL
    
    metrics : List: String, default=["nmi"]
        The metrics to measure on each run on each dataset. 
    
    k : Int
        The number of clusters for algorithms that take this as a parameter. BEWARE: This will be altered AFTER using algorithms that find their own k.
    
    min_pts : Int 
        The number of points for something to be a core point for dc-dist.
    
    eps : Float
        The maximal distance for something to be in the neighborhood, for algorithms like DBSCAN.
    
    visualize_results : Boolean, default=False
        TODO
    
    save_results : Boolean, default=False
        Whether to save the results or not in a CSV-file. 
    
    save_name : String, default="test"
        The name under which to save the CSV-file.
    
    plot_clusterings : Boolean, default=False
        Whether to plot the clusterings for each dataset or not. TODO
        
    combine_results: Boolean, default=False
        This is used to average over all the provided datasets instead of number of runs for each dataset.
    '''
    num_runtypes = len(runtypes)
    num_datasets = len(dataset_types)
    num_metrics = len(metrics)

    datasets = [] #Save the datasets here for whatever they might otherwise be used for

    #If we want extra metrics should multiply depth by number of metrics. We want the layers for the same dataset across the metrics on top of each other.
    benchmark_results = np.zeros((num_runtypes+1, num_runtypes+1, num_datasets*num_metrics)) # Make square matrix with each layer being a separate dataset it has been run on. +1 for ground truth
    headers = np.empty((num_runtypes+1, num_datasets*num_metrics), dtype=np.dtype('U100'))
    rundata = []
    
    
    benchmark_results_2 = np.zeros((num_runtypes, num_datasets, num_metrics))

    if mcs is None:
        mcs = 2
    if num_features is None:
        num_features = 2

    #TODO: add possibility to use plot_embedding that will pop up each time a set of runs has finished to visually compare clusterings.
    #TODO: add control over seeding for reproducibility 
    for d, dataset_type in enumerate(dataset_types):
        comparison_matrix = np.zeros((num_runtypes+1, num_runtypes+1, num_runs, num_metrics)) #A num_runs layer 2d matrix for each metric

        comparison_column = np.zeros((1, num_runtypes, num_runs, num_metrics))


        for i in range(num_runs):
            curr_k = k #Reset k between each run
            points, ground_truths = create_dataset(num_points, dataset_type, num_features=num_features) #Move this outside num_runs if should be same across runs
            if len(ground_truths) == len(points):
                nclusters = len(np.unique(ground_truths))-1 if -1 in ground_truths else len(np.unique(ground_truths))
            else:
                nclusters = k

            
            datasets.append(points) #Save the points generated for visualization later
            n = points.shape[0]

            curr_labels = np.zeros((num_runtypes+1, n))
            curr_labels[num_runtypes] = ground_truths

            after_hdb = False
            plot_cluster_details = []
            for r, runtype in enumerate(runtypes):
                if runtype == "HDBSCAN" or runtype == "DBSCAN" or runtype == "HDBSCAN_NEW":
                    after_hdb = True #If algo run after HDB the K could be variable and should not be part of the header (if multiple runs)
                labels, curr_k, used_min_pts, used_eps = benchmark_single(points, runtype, curr_k, min_pts, eps, mcs=mcs, gtk = nclusters)
                #Add the output labels to the collection of labels for this current dataset
                curr_labels[r] = labels
                header = ""
                plot_cluster_details.append(runtype+"_k"+str(curr_k)+"_pts"+str(used_min_pts)+"_e"+str(used_eps))
                if i == 0 and num_runs == 1:
                    #Only 1 run, can always just put the k
                    header = runtype+"_"+str(curr_k)+"_"+str(used_min_pts)+"_"+str(used_eps)
                elif i == 0:
                    #Multiple runs, only put k if before DB or HDB as they alter the K (and we take averages  in the metric output)
                    if not after_hdb:
                        header = runtype+"_k"+str(curr_k)+"_pts"+str(used_min_pts)+"_e"+str(used_eps)
                    else:
                        header = runtype+"_kvar"+"_pts"+str(used_min_pts)+"_e"+str(used_eps)
                if i == 0:
                    for m in range(num_metrics):
                        headers[r, num_metrics*d+m] = header
                        headers[num_runtypes, num_metrics*d+m] = "Ground Truth k"+str(len(np.unique(ground_truths))) #Put ground truth as last element in square
            plot_cluster_details.append("Ground Truth k"+str(len(np.unique(ground_truths))))
            #Do comparison between the different algorithms TODO: Should run this for each metric
            for m, metric in enumerate(metrics):          
                comparison_matrix[:,:,i,m] = metric_matrix(curr_labels, metric)
                
                comparison_column[0,:,i,m] = metric_column(ground_truths, curr_labels[:-1], metric)
                

            if plot_clusterings:
                #TODO: Make it possible to give a plot_embedding header
                plot_embedding(points, list(curr_labels), plot_cluster_details, centers=None, dot_scale=0.8, main_title=dataset_type + " with "+ str(num_points) + " points")

    
            
        for m, metric in enumerate(metrics):
            averaged_comparisons = np.mean(comparison_matrix[:,:,:,m], axis=2)
            benchmark_results[:,:,d*num_metrics+m] = averaged_comparisons #Insert the averaged values into the benchmark results
            #Create metadata header for each matrix
            rundata.append("d:" + dataset_type + ",p:"+ str(num_points) + ",r:"+ str(num_runs)+ ",m:"+ metric)
            averaged_column_comparisons = np.mean(comparison_column[:,:,:,m], axis=2)
            print("avg comps:", averaged_column_comparisons)
            
            benchmark_results_2[:, d, m] = averaged_column_comparisons[0]


    if save_results:
        results_to_csv(benchmark_results, headers, rundata, save_name)
    if visualize_results:
        display_results_2(benchmark_results_2, runtypes, dataset_types, metrics)
        display_results(benchmark_results, headers, rundata, dataset_types, metrics) #Should get as input the actual datasets to display the clusterings
    return

def results_to_csv(results, headers, rundata, file_name):
    with open("savefiles/benchmarks/"+file_name+".csv", "w", newline='') as res_file:
        writer = csv.writer(res_file, delimiter=";")

        for i in range(results.shape[2]): #For each layer of the results
            writer.writerow([rundata[i]])
            writer.writerow([""]+list(headers[:,i])) #Write the column header of the current layer for the current dataset

            for r, row in enumerate(results[:,:,i]): #For each of the rows in the current 2D layer
                writer.writerow([headers[r,i]] + list(row))
            writer.writerow([""]) #Empty row to separate results

        #reshaped_results = results.transpose((0,2,1)).reshape(results.shape[0], -1) #Take all layers and put beside each other from left to right, leftmost being topmost

    return


def display_results(results, headers, rundata, dataset_types, metrics):
    '''
    Displays the comparative cluster metrics in heatmap grids. 
    The grid of heatmaps is num_datasets x num_metrics big

    '''
    num_datasets = len(dataset_types)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_datasets, num_metrics, squeeze=False)
    for d in range(num_datasets):
        for m in range(num_metrics):
            ax = axes[d,m]
            curr_layer = num_metrics*d+m
            curr_map = results[:,:,curr_layer]
            curr_header = headers[:,curr_layer]
            curr_header = ["("+str(i)+") "+header for i,header in enumerate(curr_header)]
            im = ax.imshow(curr_map, cmap='viridis', interpolation=None)
            ax.set_title(rundata[curr_layer])
            ax.set_yticks(range(len(curr_header)))
            ax.set_xticks(range(len(curr_header)))
            ax.set_yticklabels(curr_header)
            fig.colorbar(im, ax=ax)

            for (j,i),label in np.ndenumerate(curr_map):
                ax.text(i,j,np.round(label, 2),ha='center',va='center', fontsize='small')



    plt.tight_layout()
    plt.show()
    return


def display_results_2(results, runtypes, dataset_types, metrics):
    '''
    Displays cluster metrics compared to ground truth in heatmap grid. 
    One big heatmap per metric.

    '''
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics)
    axes = np.atleast_1d(axes)  # Ensure axes is always treated as an array

    for m in range(num_metrics):
        ax = axes[m]
        curr_map = results[:,:,m]
        im = ax.imshow(curr_map, cmap='Blues', interpolation=None,aspect="0.3", vmin=0, vmax=1)
        ax.set_title(metrics[m])
        print("shapes", results.shape[0], results.shape[1])
        ax.set_yticks(range(results.shape[0]))
        ax.set_xticks(range(results.shape[1]))
        ax.set_yticklabels(runtypes)
        ax.set_xticklabels(dataset_types)
        fig.colorbar(im, ax=ax)

        for (j,i),label in np.ndenumerate(curr_map):
            ax.text(i,j,np.round(label, 2),ha='center',va='center', fontsize='small')



    plt.tight_layout()
    plt.show()
    return



def metric_matrix(label_results, metric="nmi"):
    '''
    Creates a matrix of size num_labellings x num_labellings computing the NMI between each. 

    Parameters
    ----------

    label_results : Array
        Array of the output clustering labels for each method run on a given dataset. (n x m) where n is the number of algorithms and m is the number of points in the dataset.
    
    metric: String, default="nmi"
        The metric to compare each pair-combination of labels on. 
        Options: "nmi", "ari", "ami"

    
    '''
    n = label_results.shape[0]
    comparison_matrix = np.zeros((n,n))
    for i, labels1 in enumerate(label_results):
        for j, labels2 in enumerate(label_results):
            if metric == "nmi": #Normalized mutual information
                comparison_matrix[i,j] = nmi(labels1, labels2)
            elif metric == "ari": # Adjusted rand index
                comparison_matrix[i,j] = ari(labels1, labels2)
            elif metric == "ami":
                comparison_matrix[i,j] = ami(labels1, labels2)
            elif metric == "test":
                comparison_matrix[i,j] = 100
            #Currently should probably just be NMI: It is symmetric. 
    return comparison_matrix

def metric_column(ground_truth, label_results, metric="ari"):
    n = label_results.shape[0]
    comparison_column = np.zeros((1,n))
    print("ground truth:", ground_truth)
    print("label_results", label_results)
    for i, labels1 in enumerate(label_results):
        if metric == "nmi": #Normalized mutual information
            comparison_column[0,i] = nmi(ground_truth, labels1)
        elif metric == "ari": # Adjusted rand index
            comparison_column[0,i] = ari(ground_truth, labels1)
        elif metric == "ami":
            comparison_column[0,i] = ami(ground_truth, labels1)
        elif metric == "test":
            comparison_column[0,i] = 100
            #Currently should probably just be NMI: It is symmetric. 
    return comparison_column

def benchmark_single(points, runtype, k, min_pts, eps, mcs, gtk):
    '''
    Runs the provided runtype on points and returns the k, min_pts and eps used in the algorithm and the resulting labels.
    '''
    labels = None
    used_min_pts = 0
    used_eps = 0
    if runtype == "HDBSCAN":
        hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples = min_pts)
        hdbscan.fit(points)
        labels = hdbscan.labels_
        k = len(np.unique(labels))
        if np.isin(-1, labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1
        used_min_pts = min_pts
    elif runtype == "HDBSCAN_NEW":
        hdbscan_new = HDBSCANNary(min_pts = min_pts, min_cluster_size=mcs)
        hdbscan_new.fit(points)
        labels = hdbscan_new.labels_
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

    elif runtype == "KCENTER":
        n = points.shape[0]
        placeholder = np.zeros(n)
        root, dc_dists = make_tree(points, placeholder, min_points=min_pts)
        labels = dc_clustering(root, len(points), k=k, min_points=min_pts, with_noise=False)
    
    elif runtype == "KMEANS": #SKLEARN Kmeans
        kmeans = KMeans(n_clusters=gtk, n_init="auto")
        kmeans.fit(points)
        labels = kmeans.labels_
    elif runtype == "KMEDOIDS":
        kmedoids = KMedoids(n_clusters=gtk, random_state=0).fit(points)
        labels = kmedoids.labels_

    elif runtype == "DCKMEANS":
        dckmeans = DCKCentroidsNary(k=k, min_pts=min_pts, loss="kmeans", noise_mode="none")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts
    elif runtype == "DCKMEANS_MED": #NOTE GTK
        dckmeans = DCKCentroidsNary(k=gtk, min_pts=min_pts, loss="kmeans", noise_mode="medium")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts
    elif runtype == "DCKMEANS_FULL":
        dckmeans = DCKCentroidsNary(k=k, min_pts=min_pts, loss="kmeans", noise_mode="full")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts

    elif runtype == "DCKMEDIAN":
        dckmedian = DCKCentroidsNary(k=k, min_pts=min_pts, loss="kmedian", noise_mode="none")
        dckmedian.fit(points)
        labels = dckmedian.labels_
        used_min_pts = min_pts
    elif runtype == "DCKMEDIAN_MED": #NOTE GTK
        dckmeans = DCKCentroidsNary(k=gtk, min_pts=min_pts, loss="kmedian", noise_mode="medium")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts
    elif runtype == "DCKMEDIAN_FULL":
        dckmeans = DCKCentroidsNary(k=k, min_pts=min_pts, loss="kmedian", noise_mode="full")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts


    elif runtype == "DCKMEANS_ELBOW":
        nk = len(points)
        dckmeans = DCKCentroidsNary(k=nk, min_pts=min_pts, loss="kmeans", noise_mode="none")
        dckmeans.fit(points)
        costs = dckmeans.cost_decreases
        costs = costs[1:] #Remove the first np.inf
        k_values = list(range(2, len(costs) + 2))  # Assuming k starts at 2 and increments by 1
        print("costs:", costs)
        kn = KneeLocator(k_values, costs, curve='convex', direction='decreasing')
        print("chosen k:", kn.knee)
        elbow_k = kn.knee
        dckmeans_elbow = DCKCentroidsNary(k=elbow_k, min_pts=min_pts, loss="kmeans", noise_mode="none")
        dckmeans_elbow.fit(points)
        labels = dckmeans_elbow.labels_
        used_min_pts = min_pts

    elif runtype == "DCKMEDIAN_ELBOW":
        nk = len(points)
        dckmedian = DCKCentroidsNary(k=nk, min_pts=min_pts, loss="kmedian", noise_mode="none")
        dckmedian.fit(points)
        costs = dckmedian.cost_decreases
        costs = costs[1:] #Remove the first np.inf
        k_values = list(range(2, len(costs) + 2))  # Assuming k starts at 2 and increments by 1
        kn = KneeLocator(k_values, costs, curve='convex', direction='decreasing')
        elbow_k = kn.knee
        print("elbow k:", elbow_k)
        dckmedian_elbow = DCKCentroidsNary(k=elbow_k, min_pts=min_pts, loss="kmedian", noise_mode="none")
        dckmedian_elbow.fit(points)
        labels = dckmedian_elbow.labels_
        used_min_pts = min_pts

    elif runtype == "DCKMEANS_GTK":
        dckmeans = DCKCentroidsNary(k=gtk, min_pts=min_pts, loss="kmeans", noise_mode="none")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts
    
    elif runtype == "DCKMEDIAN_GTK":
        dckmedian = DCKCentroidsNary(k=gtk, min_pts=min_pts, loss="kmedian", noise_mode="none")
        dckmedian.fit(points)
        labels = dckmedian.labels_
        used_min_pts = min_pts


    elif runtype == "DCKMEANS_GTK10":
        dckmeans = DCKCentroidsNary(k=gtk, min_pts=10, loss="kmeans", noise_mode="none")
        dckmeans.fit(points)
        labels = dckmeans.labels_
        used_min_pts = min_pts
    
    elif runtype == "DCKMEDIAN_GTK10":
        dckmedian = DCKCentroidsNary(k=gtk, min_pts=10, loss="kmedian", noise_mode="none")
        dckmedian.fit(points)
        labels = dckmedian.labels_
        used_min_pts = min_pts

    elif runtype == "HDBSCAN_o1":
        hdbscan_new = HDBSCANNary(min_pts = min_pts, min_cluster_size=mcs)
        hdbscan_new.fit(points, objective="simple")
        labels = hdbscan_new.labels_
        k = len(np.unique(labels))
        if np.isin(-1, labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1
        used_min_pts = min_pts
    elif runtype == "HDBSCAN_o2":
        hdbscan_new = HDBSCANNary(min_pts = min_pts, min_cluster_size=mcs)
        hdbscan_new.fit(points, objective="simple2")
        labels = hdbscan_new.labels_
        k = len(np.unique(labels))
        if np.isin(-1, labels) and k != 1: #Should not count noise labels as a set of labels
            k -= 1
        used_min_pts = min_pts
    else:
        raise AssertionError("runtype", runtype, "does not exist...")
    return labels, k, used_min_pts, used_eps




if __name__ == "__main__":
    #brute_force_comparision(num_points=10, min_pts=3)
    # points, labels = create_dataset(100, "coil")
    # print("points:", points[:100])
    # print("labels:", labels[:100])
    runtypes = ["DCKMEANS_ELBOW", "DCKMEANS_GTK", "DCKMEDIAN_ELBOW", "DCKMEDIAN_GTK", "DCKMEANS_MED","KCENTER","HDBSCAN"]
    runtypes2 = ["KMEANS", "KMEDOIDS", "DCKMEDIAN_GTK", "DCKMEANS_GTK","DCKMEDIAN_GTK10", "DCKMEANS_GTK10"]


    benchmark2d = ["testing","compound", "pathbased", "aggregate", "d31", "jain",  "spiral"]
    synth2d = ["blobs", "blobs2", "quantiles", "moons", "TODO" ]

    benchmark10d = ["coil", ]
    synth10d = ["blobs", "blobs2","quantiles", "swiss_rolls", "s_curve", "friedman"]

    #benchmark(dataset_types=["blobs", "blobs", "blobs", "blobs", "blobs"], num_points=20, num_runs=1, runtypes=["DCKMEDIAN_ELBOW", "DCKMEDIAN", "DCKMEDIAN", "DCKMEDIAN_GTK", "KMEANS", "KMEDOIDS"], metrics=["ari"], k=5, min_pts=5, mcs=5, eps=2, num_features=2, save_results=False, visualize_results=True, plot_clusterings=False)


    #benchmark2d - remember to set to use optimal k
    benchmark(dataset_types=benchmark2d, num_points=1000, num_runs=1, runtypes=runtypes, metrics=["ari"], k=5, min_pts=3, mcs=5, eps=2, num_features=2, save_results=False, visualize_results=True, plot_clusterings=False)

    #synth2d
    benchmark(dataset_types=synth2d, num_points=1000, num_runs=3, runtypes=runtypes, metrics=["ari"], k=3, min_pts=5, mcs=5, eps=2, num_features=2, save_results=False, visualize_results=True, plot_clusterings=False)

    #benchmark10d
    benchmark(dataset_types=benchmark10d, num_points=100, num_runs=1, runtypes=runtypes, metrics=["ari"], k=3, min_pts=5, mcs=5, eps=2, num_features=10, save_results=False, visualize_results=True, plot_clusterings=False)

    #synth10d
    benchmark(dataset_types=synth10d, num_points=1000, num_runs=3, runtypes=runtypes, metrics=["ari"], k=3, min_pts=5, mcs=5, eps=2, num_features=10, save_results=False, visualize_results=True, plot_clusterings=False)


