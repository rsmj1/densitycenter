Code to recreate the experiments from the Master's Thesis "Density-Connectivity Clustering". 

The distance metric is calculated in `distance_metric.py`. The dc-tree over which the algorithms are computed is found in `density_tree_nary.py`, while the binary legacy tree for old algorithms is found in `density_tree.py`.

The code to calculate the distance measure can be found in `distance_metric.py`. 
The main experiment script is located in `benchmark.py`.

The implementation of dc-dist k-means and k-median can be found in `kcentroids_nary.py`.
An implementation of HDBSCAN* over the dc-tree is provided in `HDBSCAN_nary.py`. While inefficient, this gives the option of easily changing the objective function and looking into the structure of the recursion over the tree.

If you would like to mess around with the clusterings, we recommend the sandbox file `cluster_sandbox.py` which has a setup for each method making it easy to "play" with.

We provide a set of visualization tools in `visualization.py`: plot_tree_v2 can be used to visualize the dc-tree structure, while other methods can be used to visualize clusterings, the complete graph over the dataset and the embedding into euclidean space given the ultrametric. 

You will have to download the coil-100 dataset from [here](https://www.kaggle.com/datasets/jessicali9530/coil100) and unpack it
into the path `data/coil-100`.


Legacy implementations from previous paper ''Connecting the Dots: density-connectivity distance unifies DBSCAN, k-center and spectral clustering.'':

The k-center clustering on the dc-dist is given in `density_tree.py` and `cluster_tree.py`. We provide an
implementation of DBSCAN\* in `DBSCAN.py`. Furthermore, our implementation of Ultrametric Spectral Clustering is given in `SpectralClustering.py`.



Feel free to email if you have any questions -- 201909451@post.au.dk
- Rasmus Jørgensen
