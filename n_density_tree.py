import numpy as np
from distance_metric import get_dc_dist_matrix
from tree_plotting import plot_tree
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(2000)

class NaryDensityTree:
    def __init__(self, dist, orig_node=None, path='', parent=None):
        self.dist = dist
        self.children = []

        self.label = None
        self.point_id = None

        #For efficient greedy
        self.best_center = None
        self.cost_decrease = None
        self.chosen = False

        #For pruning
        self.num_centers = 0
        self.center_path = False
        self.unique_center = None
        #--------------------

        if orig_node is not None:
            self.orig_node = orig_node
        else:
            self.orig_node = self
        self.path = path
        self.parent = parent



    @property
    def has_children(self):
        return len(self.children) != 0
    

    @property
    def is_leaf(self):
        return not self.has_children

    @property
    def in_pruned_tree(self):
        return self.orig_node is not None


    def __len__(self):
        return len(self.children)

def get_inds(all_dists, largest_dist):
    """
    It will usually be the case that one subtree has cardinality 1 and the other has cardinality n-1.
    So just randomly permute them to make tree plotting look balanced.
    """
    equal_inds = np.where(all_dists[0] == largest_dist)[0]
    unequal_inds = np.where(all_dists[0] != largest_dist)[0]
    if np.random.rand() < 0.5:
        left_inds = equal_inds
        right_inds = unequal_inds
    else:
        right_inds = equal_inds
        left_inds = unequal_inds
    return left_inds, right_inds

def _make_tree(all_dists, labels, point_ids, path=''):
    largest_dist = np.max(all_dists)
    root = DensityTree(largest_dist)
    root.path = path

    # TODO -- this will break if multiple copies of the same point. Need to first check for equal point
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        root.children = [root]
        return root

    left_inds, right_inds = get_inds(all_dists, largest_dist)

    left_split = all_dists[left_inds][:, left_inds]
    left_labels, left_point_ids = labels[left_inds], point_ids[left_inds]
    root.set_left_tree(_make_tree(left_split, left_labels, left_point_ids, path=path+'l'))
    root.left_tree.parent = root

    right_split = all_dists[right_inds][:, right_inds]
    right_labels, right_point_ids = labels[right_inds], point_ids[right_inds]
    root.set_right_tree(_make_tree(right_split, right_labels, right_point_ids, path=path+'r'))
    root.right_tree.parent = root

    root.count_children()
    return root

def make_tree(points, labels=None, min_points=1, point_ids=None):
    '''
    This creates the n-ary version of the dc-tree. 

    Parameters
    ----------

    points : 2D numpy Array
        The points over which to create the dc-tree.

    labels :  1D numpy Array, default=None
        The labels (if any) for the points
    
    min_points : Int, default=1
        The number of points for something to be considered a core-point. This is used for computing the mutual reachability distances used for the dc-distances.

    point_ids : 1D numpy Array, default=None
        The numbering of the points used in the leaves. 
    '''

    assert len(points.shape) == 2 #Check that we get a 2D matrix of points

    if len(np.unique(points, axis=0)) < len(points):
        raise ValueError('Currently not supported to have multiple duplicates of the same point in the dataset')
    dc_dists = get_dc_dist_matrix(
        points,
        min_points=min_points
    )

    if point_ids is None:
        point_ids = np.arange(int(dc_dists.shape[0]))

    root = _make_tree(dc_dists, labels, point_ids)


    return root, dc_dists



