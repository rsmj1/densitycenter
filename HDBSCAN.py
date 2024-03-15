import efficientdcdist.dctree as dcdist
import numpy as np
import numba
from density_tree import make_tree
'''
Current thoughts
- Construction of the hierarchy: 

'''


class HDBSCAN(object):
    '''
    This is an O(n^2) implementation of HDBSCAN* as propopsed by Campello, Moulavi and Sander. 


    The steps:
    1. Compute the Core Distances of all the points
    2. Compute the DCTree
    3. Bottom up recursion on the DCTree (technically the dendrogram) (potentially computing the stabilities bottom up as well while determining the best clusters)
    
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
        cdists = self.get_cdists(points, self.min_pts)

        #Compute the dendrogram structure, which is equivalent to the dc_tree structure and contains the same information when combined with the core distances
        dc_tree = make_tree(points, None, self.min_pts, )


        clusterings = self.compute_clustering(dc_tree, cdists)

        #Create the labellings from the clusterings

        return


    def compute_clustering(self, dc_tree, cdists, parent_dist=None):
        '''
        If a cluster has the same stability as the sum of chosen clusters below, it chooses the one above (the larger one).
        This is in line with the algorithm itself, but also helps us in the case of min_pts = 1, where we have clusters with stability 0.
        The stability of a cluster is given by: 

            sum_{x in C}(1/emin(x,C) - 1/emax(C))
        , where emax(C) is the maximal epsilon at which the cluster exists, and emin(x,C) is the level at which a point no longer is part of the cluster and is considered noise.

        '''
        if dc_tree.is_leaf:
            #Leaf node
            if self.min_cluster_size > 1:
                return [dc_tree.point_id], None#This point is considered noise
            else:
                #To compute the stability of single-node cluster we need the dc_dist in its parent. 
                return [dc_tree.point_id] ,(1/cdists[dc_tree.dist]) - (1/parent_dist) #This computes the stability of the 1-point cluster. 0 If the cdist is the same as the adjacent edge that was removed at the split.

        else:
            #Inner node
            if self.min_cluster_size > self.get_tree_size(dc_tree):
                return #This cluster is considered noise
            else:
                return
            




        return


    def cluster_stability(self, dc_tree, cdists):
        '''
        All internal nodes have an ID of none. 
        The dc_tree given here might be some sub-tree of the full "big" tree.
        
        '''
        


        return
    

    def get_tree_size(self, dc_tree, cdists):
        '''
        Computes the size of the current tree (and by extension current cluster).
        This algorithm is linear in the number of leaves.
        '''
        if dc_tree.is_leaf:
            return 1
        else:
            return self.get_tree_size(dc_tree.left_tree, cdists) + self.get_tree_size(dc_tree.right_tree, cdists)
