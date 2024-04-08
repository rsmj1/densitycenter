import efficientdcdist.dctree as dcdist
import numpy as np
import numba
from density_tree import make_tree
'''
Thoughts about HDBSCAN:

It has a funky way of letting quite noisy points be part of a cluster
'''


class HDBSCAN(object):
    '''
    This is an O(n^2) implementation of HDBSCAN* as propopsed by Campello, Moulavi and Sander. 

    The steps:
    1. Compute the Core Distances of all the points
    2. Compute the DCTree
    3. Bottom up recursion on the DCTree, computing the stability in each node.

    '''

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
        
        Same_dist is used to make note of distances where things need to be gathered
        If same_dist is false and it has gotten noise returned from lower recursive levels it means that we are done gathering nodes from same distance and need to check whether they comprise a cluster or still just noise.
        If same_dist is true and we have noise return it to the next level.
        If same_dist is true and we do not have noise, merge and return any noise from lower levels to the next.
        If we get noise from the level below it must mean that these are at the same distance, and the cluster from there should be propagated rather than including that noise in the cluster, since we don't yet know whether it's a cluster split or noise. 
        '''
        same_dist = False
        if dc_tree.dist == parent_dist:
            same_dist = True
        return_clusters = []
        return_stability = 0
        return_noise = []
        if dc_tree.is_leaf:
            if self.min_cluster_size > 1:
                #Min cluster size is 2 or more here [POTENTIAL NOISE]
                return [], 0, [dc_tree]#This point is considered noise and fell out of the current cluster at this height
            else:
                #Min cluster size is 1 here. To compute the stability of single-node cluster we need the dc_dist in its parent. 
                #TODO: If the difference is 0, then return point as noise. 
                stability = (1/cdists[dc_tree.point_id]) - (1/parent_dist)
                if stability > 0:
                    return [dc_tree], stability, [] #This computes the stability of the 1-point cluster. 0 If the cdist is the same as the adjacent edge that was removed at the split.
                else:
                    return [], 0, [dc_tree] #No need to set them here
        else:
            tree_size = self.get_tree_size(dc_tree)
            if self.min_cluster_size > tree_size:
                if same_dist:
                    return [], 0, [dc_tree]
                return [], 0, [] #Noise
            else:
                #Propagate down the last change in cdist since all points connected in the tree with same value correspond to one big split at one level below that parent dist that gets propagated.
                #TODO: We can reduce computation time by only computing a new stability when: We are at the top of a multi-split of same dc-distance. When we are below another cluster merge. 
                #So what to do: The mission is to somehow only print the real clusters and nothing in between. 
                #We do not explicitly need propagation information as the value inevitably is the same or higher at the top of the split anyways, so we cannot get the below clusters as real clusters anyways.

                left_clusters, left_stability, left_noise = self.compute_clustering(dc_tree.left_tree, cdists, dc_tree.dist)
                right_clusters, right_stability, right_noise = self.compute_clustering(dc_tree.right_tree, cdists, dc_tree.dist)

                if parent_dist is None: #Root call has no parent_dist.
                    #TODO: If it's the root we might still have a case where the point is not a true split.. Need to account for that. 
                    #Should just use the root distance itself here rather than the parent distance.
                    print("Root stability: ", left_stability + right_stability)
                    return left_clusters + right_clusters #We do not really need to do this, however the algorithm specifically specifies this as it stands.
                else:

                    total_stability = left_stability + right_stability 
                    all_clusters = left_clusters + right_clusters #append the clusters together, as there is no noise in either branch
                    total_noise = left_noise + right_noise
                    #If we have same_dist, we need to keep propagating results from below until we don't...

                    if not same_dist:
                        noise_size = self.get_noise_size(total_noise)
                        if noise_size >= self.min_cluster_size:
                            print("Gathered noise:", [point + 1 for point in self.get_noise_leaves(total_noise)])
                            #All the gathered noise will have the same height and now comprises a cluster itself
                            noise_cluster_stability = noise_size / dc_tree.dist - noise_size / parent_dist
                            print("Noise cluster stability:", noise_cluster_stability)
                            total_stability += noise_cluster_stability
                            print("Total stability:", total_stability)
                            all_clusters = all_clusters + [total_noise]
                            return all_clusters, total_stability, []
                        else:
                            new_stability = self.cluster_stability(dc_tree, parent_dist, tree_size, cdists)
                            #TODO: I need to gather together all the noise at the same distance, as they might constitute a cluster.... 
                            # print("nodes: ", np.array(self.get_leaves(dc_tree))+1)
                            # print("old below sum stability:", total_stability)
                            # print("left, right:", left_stability, right_stability)
                            # print("Own dist:", dc_tree.dist)
                            # print("parent_dist:", parent_dist)
                            # print("new stability:", new_stability)
                            
                            if new_stability >= total_stability and not same_dist: #Should be bigger than or equal to encompass that we get all the noise points added every time.
                                return [dc_tree], new_stability, []
                            else:                      
                                return all_clusters, total_stability, []
                    else:
                        return all_clusters, total_stability, total_noise


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
            return (left_size+right_size) * (1/dc_tree.dist)


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
        
    def get_noise_size(self, noise_list):
        '''
        Gets as input a list of dc_trees all comprised of noise, and outputs the total number of leaves in this list.
        '''
        size = 0
        for tree in noise_list:
            size += self.get_tree_size(tree)
        return size

    def get_noise_leaves(self, noise_list):
        leaves = []
        for tree in noise_list:
            leaves += self.get_leaves(tree)
        return leaves

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
            if isinstance(cluster, list): #A cluster comprised of gathered noise points at the same height
                for noise in cluster:
                    points = self.get_leaves(noise)
                    for point in points:
                        output_labels[point] = curr_label
                curr_label += 1
            else:
                points = self.get_leaves(cluster)
                for point in points:
                    output_labels[point] = curr_label
                curr_label += 1

        return output_labels
