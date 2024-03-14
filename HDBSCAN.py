import efficientdcdist.dctree as dcdist
import numpy as np


'''
Current thoughts
- Construction of the hierarchy: 

'''


class HDBSCAN(object):
    '''
    This is an O(n^2) implementation of HDBSCAN* as propopsed by Campello, Moulavi and Sander. 


    The steps:
    1. Compute the Core Distances of all the points
    2. 


    X. Bottom up recursion on the dendrogram (potentially computing the stabilities bottom up as well while determining the best clusters)
    
    '''


    def __init__(self, *, min_pts, min_cluster_size):
        '''
        
        Parameters
        ----------

        min_pts : Int
            The minimal amount of points for a point to be considered a core-point, including the point itself. 

        min_cluster_size : Int, default = 1
            Determines the minimal number of points that need to be density connected at a given epsilon to not be considered noise.
            If 1 points are considered noise if they are non-core objects. If 2 we consider a point noise even if it is still a core point, 
            but gets disconnected from the other points in the MST (but still has a self-edge essentially). From 3 and up, any connected components at a given level with fever than that amount of points are considered noise.

        '''
        
