import efficientdcdist.dctree as dcdist
import numpy as np
import numba
from density_tree import make_tree
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

    def __init__(self, *, min_pts, min_cluster_size=1, allow_single_cluster=False):
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
        placeholder = np.zeros(n)
        dc_tree,_ = make_tree(points, placeholder, self.min_pts, )
        clusterings = self.compute_clustering(dc_tree, cdists)
        labels = self.label_clusters(clusterings, n)

        self.labels_ = labels


    def compute_clustering(self, dc_tree, cdists, parent_dist=None):
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
        if dc_tree.is_leaf:
            if self.min_cluster_size > 1:
                #Min cluster size is 2 here
                return [], 0, []#This point is considered noise and fell out of the current cluster at this height
            else:
                #Min cluster size is 1 here. To compute the stability of single-node cluster we need the dc_dist in its parent. 
                #TODO: If the difference is 0, then return point as noise. 
                stability = (1/cdists[dc_tree.point_id]) - (1/parent_dist)
                if stability > 0:
                    return [dc_tree], stability, [dc_tree] #This computes the stability of the 1-point cluster. 0 If the cdist is the same as the adjacent edge that was removed at the split.
                else:
                    return [], 0, []
        else:
            tree_size = self.get_tree_size(dc_tree)
            if self.min_cluster_size > tree_size:
                return [], 0, [] #Noise
            else:
                #Propagate down the last change in cdist since all points connected in the tree with same value correspond to one big split at one level below that parent dist that gets propagated.
                #TODO: We can reduce computation time by only computing a new stability when: We are at the top of a multi-split of same dc-distance. When we are below another cluster merge. 
                #So what to do: The mission is to somehow only print the real clusters and nothing in between. 
                #We do not explicitly need propagation information as the value inevitably is the same or higher at the top of the split anyways, so we cannot get the below clusters as real clusters anyways.

                left_clusters, left_stability, left_last_clustering = self.compute_clustering(dc_tree.left_tree, cdists, dc_tree.dist)
                right_clusters, right_stability, right_last_clustering = self.compute_clustering(dc_tree.right_tree, cdists, dc_tree.dist)
                    
                var,bar,dsize,ssize = self.cluster_stability_experimental(dc_tree)

                print("node dist: ", np.round(dc_tree.dist,2), "size", len(self.get_leaves(dc_tree)))
                print("w/ var:", var,"bar", bar,"dists_size", dsize,"set_size", ssize)
                print("stability:", ssize/(1+var))
                print("")
                
                if parent_dist is None: #Root call has no parent_dist.
                    if not self.allow_single_cluster:
                        if len(left_clusters + right_clusters) == 1:
                            if len(left_last_clustering + right_last_clustering) == 1: #If we never saw any true cluster merge on the way up, return no clusters
                                return []
                            else:
                                return left_last_clustering + right_last_clustering
                        else:
                            return left_clusters + right_clusters #We do not really need to do this, however the algorithm specifically specifies this as it stands. TODO: Make it optional
                    else:
                        if len(left_clusters + right_clusters) == 1: #If we allow a single cluster here we handle that last point is noise, we return all points as a single cluster. 
                            return [dc_tree]
                        else: 
                            return left_clusters + right_clusters
                                 
                else:
                    total_stability = left_stability + right_stability 
                    all_clusters = left_clusters + right_clusters #append the clusters together, as there is no noise in either branch
                    new_stability = self.cluster_stability(dc_tree, parent_dist, tree_size, cdists)
                    #TODO: I need to gather together all the noise at the same distance, as they might constitute a cluster.... 
                    
                    # print("nodes: ", np.array(self.get_leaves(dc_tree))+1)
                    # print("old below sum stability:", total_stability)
                    # print("left, right:", left_stability, right_stability)
                    # print("Own dist:", dc_tree.dist)
                    # print("parent_dist:", parent_dist)
                    # print("new stability:", new_stability)


                    
                    if new_stability >= total_stability: #Should be bigger than or equal to encompass that we get all the noise points added every time.
                            return [dc_tree], new_stability, left_last_clustering + right_last_clustering
                    else:
                            if len(all_clusters) == 1:
                                return all_clusters, total_stability, left_last_clustering + right_last_clustering
                            else: 
                                return all_clusters, total_stability, all_clusters 


    def cluster_stability_experimental(self, dc_tree):
        '''
        Computes |C|/Var(D_C)
        Currently just blindly bottom up to check values.
        '''
        if dc_tree.is_leaf:
            return 0,0,0,1 #return var, bar, num_pairwise_dists, num_points
        else:
            lvar, lbar, l_dists_size, l_set_size = self.cluster_stability_experimental(dc_tree.left_tree)
            rvar, rbar, r_dists_size, r_set_size = self.cluster_stability_experimental(dc_tree.right_tree)

            new_set_size = l_set_size + r_set_size #The amount of points in the combined subtree
            
            new_var, new_bar, new_dists_size = self.merge_subtree_variance(dc_tree.dist, l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, l_set_size, r_set_size)
            return new_var, new_bar, new_dists_size, new_set_size
        
 
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
    
    def merge_subtree_variance(self, dist,  l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, lsetsize, rsetsize):
        if l_dists_size + r_dists_size != 0: 
            sub_var, sub_bar, sub_size = self.combined_variance(l_dists_size, r_dists_size, lvar, rvar, lbar, rbar)
            
            new_dists_size = lsetsize*rsetsize
            merge_var, merge_bar, merge_dists_size = self.combined_variance(sub_size, new_dists_size, sub_var, 0, sub_bar, dist)
        else: #If no dists in either distribution so far (coming from leaves)
            merge_var, merge_bar, merge_dists_size = 0, dist, 1

        return merge_var, merge_bar, merge_dists_size


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
        eminsum = 0
        if self.min_cluster_size > 1:
            eminsum = self.sub_contribution(dc_tree)
        else:
            eminsum = self.sub_contribution_1_1(dc_tree, cdists)
        return eminsum - emax
    
    def sub_contribution(self, dc_tree):
        '''
        Given a cluster C, this computes: sum_{x in C} 1 / emin(x,C).
        This is never called on a leaf / noise.
        '''

        left_size = self.get_tree_size(dc_tree.left_tree)
        right_size = self.get_tree_size(dc_tree.right_tree)
        if left_size < self.min_cluster_size and right_size < self.min_cluster_size:
            #Both sides noise
            return (left_size + right_size) * (1/dc_tree.dist)
        elif left_size < self.min_cluster_size:
            #LHS noise
            return left_size * (1/dc_tree.dist) + self.sub_contribution(dc_tree.right_tree)
        elif right_size < self.min_cluster_size:
            #RHS noise
            return right_size * (1/dc_tree.dist) + self.sub_contribution(dc_tree.left_tree)
        else:
            #Cluster merging
            #This checks if it is a true split or not!
            real_lsize = left_size - self.count_equidist(dc_tree.left_tree, dc_tree.dist)
            real_rsize = right_size - self.count_equidist(dc_tree.right_tree, dc_tree.dist)
            if real_lsize >= self.min_cluster_size and real_rsize >= self.min_cluster_size:
                return (left_size+right_size) * (1/dc_tree.dist)
            else:
                return self.sub_contribution(dc_tree.right_tree) + self.sub_contribution(dc_tree.left_tree)


    def sub_contribution_1(self, dc_tree, cdists): #Version with no noise points. If activating this, also remember to change the leaf case in compute_clustering to always return the point as a cluster even is stability is 0.
        '''
        For min_cluster_size = 1. 
        Given a cluster C, this computes: sum_{x in C} 1 / emin(x,C).
        This is never called on a leaf / noise. 
        '''

        left_size = self.get_tree_size(dc_tree.left_tree)
        right_size = self.get_tree_size(dc_tree.right_tree)
        if left_size == self.min_cluster_size and right_size == self.min_cluster_size:
            #Both sides noise
            return (1/cdists[dc_tree.left_tree.point_id]) + (1/cdists[dc_tree.right_tree.point_id])
        elif left_size == self.min_cluster_size:
            #LHS noise
            return (1/cdists[dc_tree.left_tree.point_id]) + right_size * (1/dc_tree.right_tree.dist)
        elif right_size == self.min_cluster_size:
            #RHS noise
            return (1/cdists[dc_tree.right_tree.point_id]) + left_size * (1/dc_tree.left_tree.dist)
        else:
            #Cluster merging
            return (left_size+right_size) * (1/dc_tree.dist)

    def sub_contribution_1_1(self, dc_tree, cdists): #Version with noise points.
            '''
            For min_cluster_size = 1. 
            Given a cluster C, this computes: sum_{x in C} 1 / emin(x,C).
            This is never called on a leaf / noise.
            This version should be consistent with the theory. 
            '''

            left_size = self.get_tree_size(dc_tree.left_tree)
            right_size = self.get_tree_size(dc_tree.right_tree)
            if left_size == self.min_cluster_size and right_size == self.min_cluster_size:
                #Both sides noise
                return (1/cdists[dc_tree.left_tree.point_id]) + (1/cdists[dc_tree.right_tree.point_id])
            
            elif left_size == self.min_cluster_size:
                #LHS potential noise
                if cdists[dc_tree.left_tree.point_id] != dc_tree.dist:
                    return (1/cdists[dc_tree.left_tree.point_id]) + right_size * (1/dc_tree.right_tree.dist)
                else:                                                   #Not a true split, still noise downwards
                    return (1/cdists[dc_tree.left_tree.point_id]) + self.sub_contribution_1_1(dc_tree.right_tree, cdists)
                
            elif right_size == self.min_cluster_size:
                #RHS potential noise
                if cdists[dc_tree.right_tree.point_id] != dc_tree.dist:
                    return (1/cdists[dc_tree.right_tree.point_id]) + left_size * (1/dc_tree.left_tree.dist)
                else:
                    #TODO: Check if true merge or not here as well.
                    return (1/cdists[dc_tree.right_tree.point_id]) + self.sub_contribution_1_1(dc_tree.left_tree, cdists)
                
            else:
                #Cluster merging
                return (left_size+right_size) * (1/dc_tree.dist)

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
        '''
        Given a clustering CL with |CL|=k, it labels the clusters C in CL from 0 to k-1. All points not in clusters are labelled as noise, -1.
        '''
        curr_label = 0
        output_labels = np.zeros(n) -1
        for cluster in clustering:
            points = self.get_leaves(cluster)
            for point in points:
                output_labels[point] = curr_label
            curr_label += 1

        return output_labels
    
    def count_equidist(self, dc_tree, dist):
        '''
        Counts the number of points within the given dc_tree that are at the given distance dist.
        '''
        if dc_tree.dist != dist:
            return 0
        else:
            lsize = self.get_tree_size(dc_tree.left_tree)
            rsize = self.get_tree_size(dc_tree.right_tree)
            if lsize == 1 and rsize == 1:
                return 2
            elif lsize == 1:
                return 1 + self.count_equidist(dc_tree.right_tree, dist)
            elif rsize == 1:
                return 1 + self.count_equidist(dc_tree.left_tree, dist)
            else: 
                return self.count_equidist(dc_tree.right_tree, dist) + self.count_equidist(dc_tree.left_tree, dist)
