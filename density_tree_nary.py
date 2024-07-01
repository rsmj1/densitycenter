import numpy as np
import numba as nb
from distance_metric import get_dc_dist_matrix
import sys
sys.setrecursionlimit(2000)

class NaryDensityTree:
    def __init__(self, dist, orig_node=None, parent=None, point_id=None, label=None, size=None):
        self.dist = dist
        self.children = []

        self.label = label
        self.point_id = point_id

        self.size = size

        #For efficient greedy
        self.best_center = None
        self.cost = None
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
        if subtree is None:
            return
        self.children.append(subtree)


def tree_builder(dists, labels, point_ids, parent=None):
    '''
    Main method for building the n-ary dc-tree. Recursively finds the next splits at each level, building the tree in recursive manner.
    It gets the next split-values. If the group is a leaf group it makes a "leaf tree" for each, otherwise it recurses on the group.

    Parameters
    ----------
    dists : 2D numpy array
        The distance matrix of the current set of data points the function is called on.

    labels : 1D numpy array
        The potential labels of the points.

    point_ids : 1D numpy array
        The ids of the points. 
    '''
    largest_dist = np.max(dists[0]) #You can always find the largest distance in every row.
    root = NaryDensityTree(largest_dist, size=dists.shape[0], parent=parent)
    
    split_groups = get_next_split(dists, largest_dist) #Get the split structure for this level of the tree

    for group in split_groups.values():
        is_leaf = group[0]
        inds = np.array(group[1])

        if is_leaf: #If the group is a "leaf group" all the indices should each be added as separate leaves from the current root.
            for i in inds:
                if labels is None:
                    root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root, size=1))
                else:
                    root.add_child(NaryDensityTree(0, point_id=point_ids[i], parent=root, label=labels[i], size=1))
        else:
            recurse_dists = dists[inds][:, inds]
            recurse_labels = labels[inds] if labels is not None else None
            recurse_point_ids =  point_ids[inds]
            root.add_child(tree_builder(recurse_dists, recurse_labels, recurse_point_ids, parent=root))

    return root
    

def get_next_split(dists, largest_dist):
    '''
    Finds the next (potentially multi) split in the tree.
    To find the split, every "side" of the split will have max distance to the same set of other nodes - the nodes are grouped by this.
    The distance matrix is first binarized based on the max distance. 
    Then a dictionary of each unique bitstring row is kept, and the nodes are grouped by their bitstrings. 
    Each group is represented by a list of [Boolean, [Nodes]], where the boolean marks whether the group is a set of leaf nodes, or further recursion should be done.
    
    '''
    n = dists.shape[0]
    split_groups = {}

    work_dists, leaf = binarize_array(dists, largest_dist) #binarize_array returns the index of a leaf within the leaf split group (if any), otherwise leaf is -1. 

    for i, row in enumerate(work_dists):
        split_group = tuple(row)
        if split_group not in split_groups: #New unseen split_group is added. 
            split_groups[split_group] = [False, []]
        if leaf == i:
            split_groups[split_group][0] = True
        split_groups[split_group][1].append(i)
    return split_groups

@nb.njit()
def binarize_array(arr, one_val):
    '''
    Based on the provided one_val, it will turn the array into a binary version with ones when dist[i,j] == one_val. 
    Leaf nodes will have ones in all entries except when i=j. In this case, these should also get a 1 added in this entry, 
    since nodes within this group exceptionally has max_val to other nodes in its group which is not the case for other split groups. 

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
            new_arr[i,i] = 1 #If we have equidistant leaves they should all get in the same group. This only happens if they also have ones for themselves - otherwise we get 01111, 10111, 11011, 11101...

    return new_arr, leaf_index


def make_n_tree(points, labels=None, min_points=1, point_ids=None):
    '''
    This creates the n-ary version of the dc-tree. 

    Parameters
    ----------
    points : 2D numpy Array
        The points over which to create the dc-tree.

    labels :  1D numpy Array, default=None
        The labels (if any) for the points (clustering labels)
    
    min_points : Int, default=1
        The number of points for something to be considered a core-point. This is used for computing the mutual reachability distances used for the dc-distances.

    point_ids : 1D numpy Array, default=None
        The numbering of the points used in the leaves. 
    '''
    assert len(points.shape) == 2 #Check that we get a 2D matrix of points

    if len(np.unique(points, axis=0)) < len(points):
        raise ValueError('Currently not supported to have multiple duplicates of the same point in the dataset')
    
    dc_dists = get_dc_dist_matrix(points, min_points=min_points)
    if point_ids is None:
        point_ids = np.arange(int(dc_dists.shape[0]))

    root = tree_builder(dc_dists, labels, point_ids, parent=None)
    return root, dc_dists


def prune_n_tree(dc_tree, min_pts, pruned_parent=None):
    '''
    Version can be used for visualization.

    Returns a copy of the tree with only the non-pruned structure left. 
    Below a cut of noise will be a leaf with a -2 point_id label.
    For something that becomes a leaf by pruning, the sub-structure under it will be reinstated. 
    '''

    #If len(dc_tree) is 1 - then it is to be pruned no matter what
    #If curr_dist is same as dist in current node, then this node is not noise. That would have been detected higher up in the tree if it was.
    if dc_tree.size >= min_pts:
        pruned_root = NaryDensityTree(dc_tree.dist, orig_node=dc_tree, parent=pruned_parent)

        for child in dc_tree.children:
            pruned_root.add_child(prune_n_tree(child, min_pts, pruned_root))

        if pruned_root.is_leaf: #If this node becomes a leaf in the pruned tree, we want its children back again.
            return dc_tree
        size = len(get_leaves(pruned_root))
        pruned_root.size = size
        return pruned_root
    else:
        return None
    

def get_leaves(dc_tree):
    '''
    Returns the set of ids of the leaf nodes within the given cluster.
    '''
    leaves = []
    def leaf_helper(dc_tree):
        if dc_tree.is_leaf:
            leaves.append(dc_tree.point_id)
        else:
            for child in dc_tree.children:
                leaf_helper(child)
    leaf_helper(dc_tree)
    return np.array(leaves)




def make_tree(points, min_points):
    '''
    Creates the dc-tree in O(n^2) using a direct MST construction via Prim's algorithm. Prim is faster than Kruskal for dense graphs.

    Steps:
    1. Create the mutual reachability distances:
        a. This can be done in O(n^2) if we use a kd-tree to find the k-th nearest neighbors for the cdists. 
            i. This might not be ideal for high dimensional data.
        b. When we have the cdists, we can get the mutual reachability distances easily in O(n^2)
    2. Create the MST
        a. Use Prim's algorithm as we are working with a complete graph, which has a complexity of O(n^2) (or in graph terms O(V^2)) if we provide an adjancency matrix, which is exactly what we output from previous mut reach step.
        b. Sort the MST edges - there are O(n) edges, so O(n*log(n)).
    3. Create the DC-tree from this.

    '''

    complete_graph_mut_reach = get_mutual_reachability_dists(points, min_points)



    return

    










#Takes as input the euclidean distance matrix D
def get_reach_dists(D, min_points, num_points):
    # Get reachability for each point with respect to min_points parameter
    reach_dists = np.sort(D, axis=1)
    reach_dists = reach_dists[:, min_points - 1] #These are the core-distances for each point.

    # Make into an NxN matrix
    reach_dists_i, reach_dists_j = np.meshgrid(reach_dists, reach_dists)

    # Take max of reach_i, D_ij, reach_j
    D = np.stack([D, reach_dists_i, reach_dists_j], axis=-1)
    D = np.max(D, axis=-1)

    # Zero out the diagonal so that it's a distance metric
    diag_mask = np.ones([num_points, num_points]) - np.eye(num_points)
    D *= diag_mask
    return D

def get_mutual_reachability_dists(points, min_points=5, **kwargs):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    @nb.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points):
        for i in nb.prange(num_points):
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

    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)
    if min_points > 1:
        if min_points > num_points:
            raise ValueError('Min points cannot exceed the size of the dataset')
    D = get_reach_dists(D, min_points, num_points)
    return D





