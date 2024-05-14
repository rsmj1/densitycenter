import efficientdcdist.dctree as dcdist
import numpy as np
import numba
from n_density_tree import make_n_tree, get_leaves
'''
Thoughts about HDBSCAN:

It has a funky way of letting quite noisy points be part of a cluster

Implementation observations:
- If we treat the root cluster separately we do get some issues in terms of noise clusters still being a thing, "ish". 
- The reason why this is only an issue when treating the root path separately is because the noise clusters merged with the closest cluster will always give a higher stability. 
- We still have the issue that these noise clusters will potentially fuck up the stability higher up in the tree if we don't specifically search for them.
- So no matter what I need to fix the stability computation... If there is a split with something at the same dist in one of the subtree, we need to check whether there is actually any connected component to that side of the split...


'''


class HDBSCAN(object):
    '''
    This is an O(n^2) implementation of HDBSCAN* as propopsed by Campello, Moulavi and Sander. 

    The steps:
    1. Compute the Core Distances of all the points
    2. Compute the DCTree
    3. Bottom up recursion on the DCTree, computing the stability in each node.



    '''

    def __init__(self, *, min_pts, min_cluster_size=1, allow_single_cluster=False, tree=None):
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
        self.allow_single_cluster = allow_single_cluster

        self.cdists = None
        #Temporary var for visualization
        self.extra_annotations = []
        self.tree = tree #Provided tree to ensure same order traversal as visualization

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
        '''
        1. Compute the core distances for each point
        2. Compute the dendrogram structure, which is equivalent to the dc_tree structure and contains the same information when combined with the core distances
        3. Compute the clustering over the dc_tree. 
        4. Label the clustering.
        '''
        n = points.shape[0]
        if n < self.min_pts or n < self.min_cluster_size:
            raise AssertionError("n should be larger than both min_pts and min_cluster size...")

        cdists = self.get_cdists(points, self.min_pts)
        self.cdists = cdists
        placeholder = np.zeros(n)
        if self.tree is None: #Important this assumes that the tree provided is for the correct points!!
            dc_tree,_ = make_n_tree(points, placeholder, self.min_pts, )
        else:
            dc_tree = self.tree

        clusterings = self.compute_clustering(dc_tree, cdists)
        labels = self.label_clusters(clusterings, n)

        self.labels_ = labels


    def compute_clustering(self, dc_tree, cdists, merge_above=True):
        '''
        Main HDBSCAN computation algorithm over the n-ary dc-tree.

        Parameters
        ----------

        dc_tree : NaryDensityTree
            The dc-tree over the set of points the HDBSCAN algorithm is run on.
        cdists : np.array
            An array of the core-distances between all of the points. 
        merge_above : Boolean, default=True
            This indicates whether we have a cluster split from above / a merge from below, meaning that stability computations should be made to evaluate the cluster at this level if True.
        '''
        if self.min_cluster_size > dc_tree.size:
            #Temp code for stability annotations:
            var, bar, dsize = self.cluster_statistics(dc_tree)
            self.extra_annotations.append(var)


            #print("leaf, returning noise", self.get_leaves(dc_tree)+1)
            return [], 0 #Noise
        else:
            #Temp code for stability annotations:
            var, bar, dsize = self.cluster_statistics(dc_tree)
            self.extra_annotations.append(var)

            #Real code from here and below
            total_stability = 0
            below_clusters = []
            split_size = self.split_size(dc_tree, self.min_cluster_size)
            
            for child in dc_tree.children:
                if child.size >= self.min_cluster_size and split_size >= 2: #Currently we do not allow multiple points at the same height (as leaves) to constitute a split.
                    clusters, stability = self.compute_clustering(child, cdists, True)
                else:
                    clusters, stability = self.compute_clustering(child, cdists, False)

                total_stability += stability
                below_clusters += clusters


            if merge_above: #When we have a merge above, we know that this current subtree in its entirety might constitute a cluster depending on stability computations within it.                
                if dc_tree.parent is None: #Root call
                    #print("Root call check!")
                    if len(below_clusters) <= 1:
                        if self.allow_single_cluster:
                            return [dc_tree]
                        else:
                            return []
                    else:
                        return below_clusters
                    
                else: #We compute the stability only here - all our stability measurements work bottom up - so even for the "new" stability measurement, we can just continue working up from the previous "checkpoint".
                    new_stability = self.cluster_stability(dc_tree, dc_tree.parent.dist, dc_tree.size, cdists)

                    if new_stability >= total_stability:
                        if new_stability == 0:#This means that it is a leaf with same cdist as the distance above
                            return [], 0
                        
                        return [dc_tree], new_stability
                    else:
                        return below_clusters, total_stability
            else:
                return below_clusters, total_stability
                


    def split_size(self, dc_tree, min_cluster_size):
        split_size = 0
        if min_cluster_size != 1:
            for child in dc_tree.children:
                if child.size >= min_cluster_size: #Currently we do not allow multiple points at the same height (as leaves) to constitute a split.
                        split_size += 1
            return split_size
        else:
            for child in dc_tree.children:
                if child.is_leaf and self.cdists[child.point_id] == dc_tree.dist: #It is not a split if the point's self edge and adjacent edge are removed at the same time.
                    continue
                split_size += 1
            return split_size

    def cluster_stability(self, dc_tree, parent_dist, tree_size, cdists):
        '''
        All internal nodes have an ID of none. 
        The dc_tree given here might be some sub-tree of the full "big" tree.
        
        A point falls out of a cluster when new ones start existing that they become part of or when they become noise, whichever comes first top down.
        The stability of a cluster is given by: 

            sum_{x in C}(1/emin(x,C) - 1/emax(C))
        , where emax(C) is the maximal epsilon at which the cluster exists, and emin(x,C) is the level at which a point no longer is part of the cluster and is considered noise.
        Can be rewritten to (sum_{x in C} 1 / emin(x,C)) - |C|/emax(C), which is what is computed here.
        Emin is the level at which each point became part of noise
        Recurse and at each level if one side is noise return that value, for the other side continue recursing.
        Stop the recursion when two clusters merge and return that level value for all nodes below.
        
        '''
        emax = tree_size/parent_dist
        eminsum = self.sub_contribution(dc_tree, cdists)
        return eminsum - emax
    
    def sub_contribution(self, dc_tree, cdists):
        '''
        Given a cluster C, this computes: sum_{x in C} 1 / emin(x,C).
        This is never called on a leaf / noise.
        '''
        if dc_tree.is_leaf:
            if self.min_cluster_size == 1:
                stability = (1/cdists[dc_tree.point_id])
                return stability
            else:
                return 0
        split_size = self.split_size(dc_tree, self.min_cluster_size)
        if split_size >= 2:
            return dc_tree.size /dc_tree.dist
        else:
            total_stability = 0
            for child in dc_tree.children:
                if child.size >= self.min_cluster_size:
                    total_stability += self.sub_contribution(child, cdists)
                else:
                    total_stability += child.size/dc_tree.dist
            return total_stability

        
    
    
    def label_clusters(self, clustering, n):
        '''
        Given a clustering CL with |CL|=k, it labels the clusters C in CL from 0 to k-1. All points not in clusters are labelled as noise, -1.
        '''
        curr_label = 0
        output_labels = np.zeros(n) -1
        for cluster in clustering:
            points = get_leaves(cluster)
            for point in points:
                output_labels[point] = curr_label
            curr_label += 1

        return output_labels



            ###############################################
            ############ NEW STABILITY STUFF ##############

    def cluster_statistics(self, dc_tree):
        '''
        Computes the variance of the dc-distance matrix of the set of nodes and all subsets of it bottom up.
        Returns the variance, mean and size of the distance matrix (lower triangular).
        '''
        if dc_tree.is_leaf:
            return 0,0,0 #return var, bar, num_pairwise_dists
        else:
            total_var, total_bar, total_dists_size = self.cluster_statistics(dc_tree.children[0])
            total_tree_size = dc_tree.children[0].size
            for child in dc_tree.children[1:]:
                new_var, new_bar, new_dists_size = self.cluster_statistics(child)
                total_var, total_bar, total_dists_size = self.merge_subtree_variance(dc_tree.dist, total_dists_size, new_dists_size, total_var, new_var, total_bar, new_bar, total_tree_size, child.size)
                total_tree_size += child.size
            
            return total_var, total_bar, total_dists_size
        
 
    def merge_subtree_variance(self, dist,  l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, lsetsize, rsetsize):
        if l_dists_size + r_dists_size != 0: 
            sub_var, sub_bar, sub_size = self.combined_variance(l_dists_size, r_dists_size, lvar, rvar, lbar, rbar)
            
            new_dists_size = lsetsize*rsetsize
            merge_var, merge_bar, merge_dists_size = self.combined_variance(sub_size, new_dists_size, sub_var, 0, sub_bar, dist)
        else: #If no dists in either distribution so far (coming from leaves)
            merge_var, merge_bar, merge_dists_size = 0, dist, 1

        return merge_var, merge_bar, merge_dists_size

    def combined_variance(self, n,m,xvar,yvar,xbar,ybar):
        '''
        https://stats.stackexchange.com/questions/557469/difference-between-pooled-variance-and-combined-variance

        Returns the combined size, combined mean and combined variance
        '''
        nc = n+m
        xc = self.combined_mean(n,m,xbar,ybar)
        vc = (n*(xvar+(xbar-xc)**2)+m*(yvar+(ybar-xc)**2))/(nc)
        return vc, xc, nc

    def combined_mean(self, n,m,xbar,ybar):
        return (xbar*n+ybar*m)/(n+m)
    
    
