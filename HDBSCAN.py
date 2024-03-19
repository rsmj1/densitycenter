import efficientdcdist.dctree as dcdist
import numpy as np
import numba
from density_tree import make_tree
'''
Current thoughts
- If I prune the tree I get issues while trying to compute tree sizes and so on...
- You will always start with noise from the bottom, however, as you get a good stability the noise will be moved into clusters
- New cluster starts either when enough points gathered to be above threshold and none of below are clusters or when two below clusters end
- Cluster ends when merged with another big enough cluster
- All noise points that fall out of the cluster: It does not matter when it does, however, note that 
'''


class HDBSCAN(object):
    '''
    This is an O(n^2) implementation of HDBSCAN* as propopsed by Campello, Moulavi and Sander. 


    The steps:
    1. Compute the Core Distances of all the points
    2. Compute the DCTree
    3. Bottom up recursion on the DCTree (technically the dendrogram) (potentially computing the stabilities bottom up as well while determining the best clusters)
    
    '''


    #class SubCluster(object):

    #    def __init__(self, clusters, noise, ):




    def __init__(self, *, min_pts, min_cluster_size=1):
        '''
        
        Parameters
        ----------

        min_pts : Int
            The minimal amount of points for a point to be considered a core-point, including the point itself. 

        min_cluster_size : Int, default = 1
            Determines the minimal number of points that need to be density connected at a given epsilon to not be considered noise.
            If 1 points are considered noise if they are non-core objects. If 2 we consider a point noise even if it is still a core point, 
            but gets disconnected from the other points in the MST (but still has a self-edge essentially). From 3 and up, any connected components at a given level with fever than that amount of points are considered noise.
            With 1 there are no noise points output, as all noise points techically at the split point are a cluster with stability 0 (or more if a core-point for longer).
            Points will still be considered noise with respect to a cluster at some epsilon level, however. 
        '''
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        
    

    def get_cdists(self, points, min_pts):
        '''
        Computes the core distances of a set of points, given a min_pts.
        '''
        num_points = points.shape[0]
        dim = int(points.shape[1])

        D = np.zeros([num_points, num_points])
        D = self.get_dist_matrix(points, D, dim, num_points)

        cdists = np.sort(D, axis=1)
        cdists = cdists[:, min_pts - 1] #These are the core-distances for each point.
        #print("cdists:", cdists)
        return cdists

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points):
        '''
        Returns the Euclidean distance matrix of a 2D set of points. 

        Parameters
        ----------

        points : n x m 2D numpy array
        D : empty n x n numpy array
        dim : m
        num_points : n
        '''
        for i in numba.prange(num_points):
            x = points[i]
            for j in range(i+1, num_points):
                y = points[j]
                dist = 0
                for d in range(dim):
                    dist += (x[d] - y[d]) ** 2
                dist = np.sqrt(dist)
                D[i, j] = dist
                D[j, i] = dist
        return D
    

    def fit(self, points):
        #Compute the core distances for each point
        n = points.shape[0]
        cdists = self.get_cdists(points, self.min_pts)
        placeholder = np.zeros(n)
        #Compute the dendrogram structure, which is equivalent to the dc_tree structure and contains the same information when combined with the core distances
        dc_tree,_ = make_tree(points, placeholder, self.min_pts, )

        #Compute the actual clustering
        clusterings = self.compute_clustering(dc_tree, cdists)

        #Create the labellings from the clusterings
        labels = self.label_clusters(clusterings, n)

        self.labels_ = labels


    def compute_clustering(self, dc_tree, cdists, parent_dist=None, is_propagated=False):
        '''
        If a cluster has the same stability as the sum of chosen clusters below, it chooses the one above (the larger one).
        This is in line with the algorithm itself, but also helps us in the case of min_pts = 1, where we have clusters with stability 0.
        The stability of a cluster is given by: 

            sum_{x in C}(1/emin(x,C) - 1/emax(C))
        , where emax(C) is the maximal epsilon at which the cluster exists, and emin(x,C) is the level at which a point no longer is part of the cluster and is considered noise.

        The clusters are propagated bottom up as lists of lists. Each particle in this representation is determined by the size of min_cluster_size.
        The cluster representation returned is ([cluster1,...,clusterk], stability_sum), where clusteri = [atom1,...,atomt], where atomj = (tree_pointer, tree_size, breakoff_dist).
        Parent_dist is the maximal point at which the cluster exists.
        '''
        #print("current nodes:", self.get_leaves(dc_tree))

        if dc_tree.is_leaf:
            #Leaf node
            if self.min_cluster_size > 1:
                #Min cluster size is 2 here
                return [], [[(dc_tree, 1, parent_dist)]], 0#This point is considered noise and fell out of the current cluster at this height
            else:
                #To compute the stability of single-node cluster we need the dc_dist in its parent. 
                #Min cluster size is 1 here
                return [[(dc_tree, 1, parent_dist)]], [], (1/cdists[dc_tree.point_id]) - (1/parent_dist) #This computes the stability of the 1-point cluster. 0 If the cdist is the same as the adjacent edge that was removed at the split.

        else:
            #Inner node
            tree_size = self.get_tree_size(dc_tree)
            #print("tree_size, mcs:", tree_size, self.min_cluster_size)
            if self.min_cluster_size > tree_size:
                #print("returning current nodes: ", self.get_leaves(dc_tree))
                return [], [[(dc_tree, tree_size, parent_dist)]], 0 #This cluster is considered noise when working with the given cluster_size
            else:
                #Propagate down the last change in cdist since all points connected in the tree with same value correspond to one big split at one level below that parent dist that gets propagated.
                recursion_dist_left = dc_tree.dist
                recursion_dist_right = dc_tree.dist
                propagate_left, propagate_right = False, False
                if dc_tree.left_tree.dist == dc_tree.dist:
                    recursion_dist_left = parent_dist
                    propagate_left = True
                if dc_tree.right_tree.dist == dc_tree.dist:
                    recursion_dist_right = parent_dist
                    propagate_right = True
                left_clusters, left_noise, left_stability = self.compute_clustering(dc_tree.left_tree, cdists, recursion_dist_left, propagate_left)
                right_clusters, right_noise, right_stability = self.compute_clustering(dc_tree.right_tree, cdists, recursion_dist_right, propagate_right)


                if parent_dist is None: #Root call has no parent_dist.
                    return left_clusters + right_clusters
                else:
                    #Not the root node

                    if is_propagated: #Even if both are, we will always have that the sum of the sizes of the two sub-parts will be higher than min_cluster size due to how the recursion works and where it stops.
                        #If it is propagated and merges two real clusters, propagate these new clusters up, but do not update the levels at which the noise falls off, as the points above will need these same calculations
                        #If it does not merge two clusters then still don't update the levels at which noise falls off other than this particular noise point at this level.
                        print("TODO")
                        #We need to treat all points with the same cdists that are connected as single point on the way up as well...
                        #Also we need to propagate up the fact that two clusters have been merged, since we need to know evne if the last part of the merging is with noise that we need to update distances.
                    else:
                        print("TODO")
                        #If noise then do not update the levels at which points fall off.
                        #If not noise then update the levels


                    total_stability = left_stability + right_stability 
                    all_clusters = left_clusters + right_clusters #append the clusters together, as there is no noise in either branch
                    all_noise = left_noise + right_noise #append the noise as well
                    new_stability = self.cluster_stability(all_clusters, all_noise, parent_dist, tree_size)
                    print("nodes: ", np.array(self.get_leaves(dc_tree))+1)
                    print("old below sum stability:", total_stability)
                    print("left:", left_stability)
                    print("right:", right_stability)
                    print("new stability:", new_stability)
                    
                    if new_stability >= total_stability: #Should be bigger than or equal to encompass that we get all the noise points added every time.
                        #print("old below sum stability:", total_stability)
                        #print("new stability:", new_stability)
                        #Make new cluster, by merging all: [cluster1, cluster2] -> [cluster1+cluster2]
                        return [sum(all_clusters+all_noise, [])],[], new_stability
                    else:
                        print("NOT BIGGER")
                        
                        return all_clusters, all_noise, total_stability



    def cluster_stability(self, sub_clusters, noise, parent_dist, tree_size):
        '''
        All internal nodes have an ID of none. 
        The dc_tree given here might be some sub-tree of the full "big" tree.
        
        TODO: When is a point considered noise? For now, I assume it is part of a noise-cluster. 
        The stability of a cluster is given by: 

            sum_{x in C}(1/emin(x,C) - 1/emax(C))
        , where emax(C) is the maximal epsilon at which the cluster exists, and emin(x,C) is the level at which a point no longer is part of the cluster and is considered noise.

        
        '''
        #Emax is the parent distance, as this is the point at which this cluster branched from the parent
        #print("tree_size:", tree_size)

        emax = 1/parent_dist
        #Emin is the level at which each point became part of noise
        eminsum = np.sum([self.sub_contribution(sub_cluster) for sub_cluster in sub_clusters])
        eminsum += np.sum([self.sub_contribution(sub_noise) for sub_noise in noise])

        
        return eminsum - tree_size * emax
    
    def sub_contribution(self, sub_cluster):
        # Each sub_cluster element is represented as a tuple (tree_pointer, number of leaves, value for each leaf).
        #[print("subvalue:", 1/atom_value) for _, atom_size, atom_value in sub_cluster]
        
        return [np.sum([atom_size/atom_value for _,atom_size,atom_value in sub_cluster])]


    def get_tree_size(self, dc_tree):
        '''
        Computes the size of the current tree (and by extension current cluster).
        This algorithm is linear in the number of leaves.
        '''
        if dc_tree.is_leaf:
            return 1
        else:
            return self.get_tree_size(dc_tree.left_tree) + self.get_tree_size(dc_tree.right_tree)
        

    def get_leaves(self, dc_tree):
        '''
        Returns the set of ids of the leaf nodes within the given cluster.
        '''
        if dc_tree.is_leaf:
            return [dc_tree.point_id]
        else:
            #print("left:", self.get_leaves(dc_tree.left_tree))
            #print("right:", self.get_tree_size(dc_tree.right_tree))
            return self.get_leaves(dc_tree.left_tree) + self.get_leaves(dc_tree.right_tree)
        

    def label_clusters(self, clustering, n):
        curr_label = 0
        output_labels = np.zeros(n) -1
        for cluster in clustering:
            for atom_tree in cluster:
                #print("atom_tree:", atom_tree)
                points = self.get_leaves(atom_tree[0])
                for point in points:
                    output_labels[point] = curr_label
            curr_label += 1

        return output_labels


