import numpy as np
from distance_metric import get_dc_dist_matrix
from tree_plotting import plot_tree
import matplotlib.pyplot as plt
import numba as nb

import sys
sys.setrecursionlimit(2000)

class NaryDensityTree:
    def __init__(self, dist, orig_node=None, parent=None, point_id=None, label=None):
        self.dist = dist
        self.children = []

        self.label = label
        self.point_id = point_id

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
    
    def add_child(self, subtree):
        self.children.append(subtree)

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

def tree_builder(dists, labels, point_ids):
    '''
    Inner nodes will have a distance while leaf nodes will have a distance of 0.

    Leaf nodes will have (n-1)*largest_dist as the sum of their row.

    '''
    print("dists:", np.round(dists,2))

    largest_dist = np.max(dists[0]) #You can always find the largest distance in every row.
    best_inds, is_leaf = get_next_inds(dists, largest_dist)
    other_inds = np.array([i for i in np.arange(dists.shape[0]) if i not in set(best_inds)])

    root = NaryDensityTree(largest_dist)

    print("best_inds:", best_inds)
    print("worst_inds:", other_inds)
    print("is_leaf?", is_leaf)

    other_recurse_dists = dists[other_inds][:, other_inds]
    other_recurse_labels = labels[other_inds] if labels is not None else None
    other_recurse_point_ids =  point_ids[other_inds]

    if is_leaf:
        for i in best_inds:
            if labels is None:
                root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root))
            else:
                root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root, label=labels[i]))
        root.add_child(tree_builder(other_recurse_dists, other_recurse_labels, other_recurse_point_ids))

    else:
        recurse_dists = dists[best_inds][:, best_inds]
        recurse_labels = labels[best_inds] if labels is not None else None
        recurse_point_ids =  point_ids[best_inds]
        root.add_child(tree_builder(other_recurse_dists, other_recurse_labels, other_recurse_point_ids))
        root.add_child(tree_builder)

    #Find the biggest distance in the matrix. Find all points (if more than one)
    return

@nb.njit()
def get_next_inds(dists, largest_dist):
    '''
    Finds the next split in the tree, returning the indices of one side of the split
    The sum is just a counter of the number of max_dists encountered. No need to sum up to introduce numerical errors in final comparison.
    
    TODO: Add support for multi splits at same height - we can detect how big the split is by the amount of unique numbers of the max dist we have in the array.
    Every set of same number of max dist will be its own subtree. 
    '''
    best_sum = 0
    best_rows = np.zeros(dists.shape[0], dtype=np.int64)
    ctr = 0
    for i in range(dists.shape[0]):
        row_sum = 0
        for j in range(dists.shape[1]):
            if dists[i,j] == largest_dist: #Only sum up the maximal distances as these are what we use to find the subtrees of the split.
                row_sum += 1
        if row_sum > best_sum:
            best_sum = row_sum
            best_rows[0] = i
            ctr = 1
        elif row_sum == best_sum:
            best_rows[ctr] = i
            ctr += 1

    #print("best_sum:", best_sum)
    #print("upper_limit:",  dists.shape[0]-1)
    
    if best_sum == dists.shape[0]-1:
        #Then they are leaves
        return best_rows[:ctr], True
    else:
        #Otherwise it is an internal node split
        return best_rows[:ctr],  False





def _make_tree(all_dists, labels, point_ids, path=''):
    largest_dist = np.max(all_dists)
    root = NaryDensityTree(largest_dist)
    root.path = path

    # TODO -- this will break if multiple copies of the same point. Need to first check for equal point
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        root.children = [root]
        return root
    print("largest_dist:", largest_dist)
    left_inds, right_inds = get_inds(all_dists, largest_dist)
    print("left_inds:", left_inds)
    print("right_inds:", right_inds)

    left_split = all_dists[left_inds][:, left_inds]
    right_split = all_dists[right_inds][:, right_inds]

    print("left_split:", np.round(left_split,2))
    print("right_split:", np.round(right_split,2))
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

    root = tree_builder(dc_dists, labels, point_ids)
    #root = _make_tree(dc_dists, labels, point_ids)


    return root, dc_dists



points = np.array([[1,2],
                    [1,4],
                    [2,3],
                    [1,1],
                    [-5,15], #5
                    [11,13],
                    [13,11],
                    [10,8],
                    [14,13],
                    [16,17], #10
                    [18,19],
                    [19,18],
                    ]
                    )
labels = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

make_tree(points, labels, min_points=3)