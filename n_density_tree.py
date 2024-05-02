import numpy as np
from distance_metric import get_dc_dist_matrix
from tree_plotting import plot_tree
import matplotlib.pyplot as plt
import numba as nb
from visualization import plot_nary_tree
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

    @property
    def num_children(self):
        return len(self.children)

    def __len__(self):
        return len(self.children)
    
    def add_child(self, subtree):
        self.children.append(subtree)



def tree_builder(dists, labels, point_ids):
    '''
    Inner nodes will have a distance while leaf nodes will have a distance of 0.

    Leaf nodes will have (n-1)*largest_dist as the sum of their row.

    '''
    #print("dists:", np.round(dists,2))

    largest_dist = np.max(dists[0]) #You can always find the largest distance in every row.
    root = NaryDensityTree(largest_dist)
    
    
    split_groups = get_next_inds(dists, largest_dist) #Get the split structure for this level of the tree

    for group in split_groups.values():
        is_leaf = group[0]
        inds = np.array(group[1])
        #print("I have indices:", inds)
        #print("leaf?", is_leaf)
        if is_leaf: #If the group is a "leaf group" all the indices should each be added as separate leaves from the current root.
            for i in inds:
                if labels is None:
                    root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root))
                else:
                    root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root, label=labels[i]))
        else:
            recurse_dists = dists[inds][:, inds]
            recurse_labels = labels[inds] if labels is not None else None
            recurse_point_ids =  point_ids[inds]
            root.add_child(tree_builder(recurse_dists, recurse_labels, recurse_point_ids))

    return root
    

def get_next_inds(dists, largest_dist):
    '''
    Finds the next split in the tree, returning the indices of one side of the split
    The sum is just a counter of the number of max_dists encountered. No need to sum up to introduce numerical errors in final comparison.
    
    TODO: Add support for multi splits at same height - we can detect how big the split is by the amount of unique numbers of the max dist we have in the array.
    Every set of same number of max dist will be its own subtree. 
    '''
    n = dists.shape[0]
    split_groups = {}

    work_dists, leaf = binarize_array(dists, largest_dist)

    #print("work_dists:", work_dists)
    for i, row in enumerate(work_dists):
        split_group = tuple(row)
        if split_group not in split_groups:
            #print("split_group:", split_group)
            split_groups[split_group] = [False, []]
        if leaf == i:
            split_groups[split_group][0] = True
        split_groups[split_group][1].append(i)
    return split_groups

@nb.njit()
def binarize_array(arr, one_val):
    '''
    Based on the provided one_val, it will turn the array into a binary version with ones for point to itself and dist==one_val.

    Also checks whether there are any points that will be leaves at this level, returning true if yes, and returning one of the points that will be a leaf. 
    '''
    new_arr = np.zeros(arr.shape, dtype=np.int64)
    leaf_index = -1
    for i in range(arr.shape[0]):
        row_sum = 0
        for j in range(arr.shape[1]):
            if arr[i,j] == one_val:
                new_arr[i,j] = 1
                row_sum += 1
        if row_sum == arr.shape[0]- 1:
            leaf_index = i
            new_arr[i,i] = 1 #If we have equidistant leaves they should all get in the same group. This only happens if they also have ones for themselves - otherwise we get 0111, 1011, 1101, 1110...

    return new_arr, leaf_index


def make_n_tree(points, labels=None, min_points=1, point_ids=None):
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


    return root, dc_dists


#root, _ = make_n_tree(points, labels, min_points=3)

#plot_nary_tree(root, labels)