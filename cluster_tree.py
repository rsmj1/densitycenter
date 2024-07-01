import numpy as np
from density_tree import DensityTree
from density_tree_nary import NaryDensityTree

class Cluster:
    def __init__(self, center, points, peak):
        self.center = center
        self.points = points
        self.peak = peak

    def __len__(self):
        return len(self.points)

def copy_tree(root, min_points, pruned_parent=None):
    '''
    Returns a copy of the tree with only the non-pruned structure left. 
    orig_node saves the old node before the modification for reinstating stuff again.

    It prunes by making "none" children from the pruned things, keeping the overalll structure otherwise.

    Example with min_pts = 3: (blue path does not mean anything)

          o
         / \
        o   o
       /     \
      o       o
     / \     / \
    o   o   o   o
                 \
                  o
                 / \
                o   o
    
    Becomes:
    
          o
         / \
        N   o
             \
              o
             / \
            N   N
       



    '''
    if len(root) >= min_points:
        pruned_root = DensityTree(root.dist, orig_node=root, path=root.path, parent=pruned_parent)
        if root.left_tree is not None:
            pruned_root.set_left_tree(copy_tree(root.left_tree, min_points, pruned_root))
        if root.right_tree is not None:
            pruned_root.set_right_tree(copy_tree(root.right_tree, min_points, pruned_root))
        pruned_root.count_children()
        return pruned_root

    return None

def prune_tree(dc_tree, min_pts, pruned_parent=None, curr_dist=None):
    '''
    Version can be used for visualization.

    Returns a copy of the tree with only the non-pruned structure left. 
    Below a cut of noise will be a leaf with a -2 point_id label.
    For something that becomes a leaf by pruning, the sub-structure under it will be reinstated. 
    '''

    #If len(dc_tree) is 1 - then it is to be pruned no matter what
    #If curr_dist is same as dist in current node, then this node is not noise. That would have been detected higher up in the tree if it was.

    if dc_tree.dist == 0.0:
        equidist = 0
    else:
        equidist = count_equidist(dc_tree, dc_tree.dist) #When counting from the root itself, it will always count that node as one, which should be subtracted
    size = len(dc_tree)
    total_size = size - equidist
    #print("curr_leaves:", np.array(get_leaves(dc_tree))+1)
    #print("size:", size, "total_size:", total_size, "curr_dist:", curr_dist, "dc_tree.dist:", dc_tree.dist)
    #print("with truthness:", (size >= min_pts and total_size != 0) or curr_dist == dc_tree.dist)
    if (size >= min_pts and total_size != 0) or (curr_dist == dc_tree.dist and total_size != 0):
        pruned_root = DensityTree(dc_tree.dist, orig_node=dc_tree, path=dc_tree.path, parent=pruned_parent)
        if dc_tree.left_tree is not None:
            pruned_root.set_left_tree(prune_tree(dc_tree.left_tree, min_pts, pruned_root, dc_tree.dist))
        if dc_tree.right_tree is not None:
            pruned_root.set_right_tree(prune_tree(dc_tree.right_tree, min_pts, pruned_root, dc_tree.dist))
        pruned_root.count_children()

        #Now reinstate things for usage and visualization
        if pruned_root.is_leaf and min_pts > 1: #If leaf - put everything back under it.
            pruned_root = dc_tree
            return pruned_root

        if pruned_root.left_tree is None:
            leaf = DensityTree(0, parent=pruned_root, orig_node = dc_tree.left_tree) 
            leaf.point_id = -2 
            pruned_root.set_left_tree(leaf)
        if pruned_root.right_tree is None:
            leaf = DensityTree(0, parent=pruned_root, orig_node=dc_tree.right_tree)
            leaf.point_id = -2
            pruned_root.set_right_tree(leaf)

        return pruned_root
    #Else:
    if size >= min_pts and curr_dist != dc_tree.dist: #The curr_dist != dc_tree.dist decides whether to call multiple nodes at same dist when from that node also other things move from there noise or not.
        #If size >= min_pts, this must be a leaf node in the pruned tree
        #This is to account for the fact that when we have multiple nodes at the same distance they might make the above node in the tree look like a leaf, although it is the root of the multiple at same distance that is the leaf of the pruned tree.
        #We then reinstate anything below this.
        return dc_tree
    
    return None  #Not a leaf - just a pruned part of the tree.




def get_leaves(dc_tree):
    '''
    Returns the set of ids of the leaf nodes within the given cluster.
    '''
    if dc_tree.is_leaf:
        return [dc_tree.point_id]
    else:
        #print("left:", self.get_leaves(dc_tree.left_tree))
        #print("right:", self.get_tree_size(dc_tree.right_tree))
        return get_leaves(dc_tree.left_tree) + get_leaves(dc_tree.right_tree)




def count_equidist(dc_tree, dist):
    '''
    Counts the number of points within the given dc_tree that are at the given distance dist.
    Cannot be called on a leaf node with distance of 0 - will fail. 
    '''
    if dc_tree.dist != dist:
        return 0
    else:
        lsize = len(dc_tree.left_tree)
        rsize = len(dc_tree.right_tree)
        if lsize == 1 and rsize == 1:
            return 2
        elif lsize == 1:
            return 1 + count_equidist(dc_tree.right_tree, dist)
        elif rsize == 1:
            return 1 + count_equidist(dc_tree.left_tree, dist)
        else: 
            return count_equidist(dc_tree.right_tree, dist) + count_equidist(dc_tree.left_tree, dist) 


def get_node(root, path):
    if path == '':
        return root
    if path[0] == 'l':
        return get_node(root.left_tree, path[1:])
    return get_node(root.right_tree, path[1:])

def get_lca_path(left, right):
    depth = 0
    while left.path[depth] == right.path[depth]:
        depth += 1
        if depth >= len(left.path) or depth >= len(right.path):
            break
    return left.path[:depth]

def merge_costs(dist, c1, c2):
    merge_right_cost = dist
    merge_left_cost = dist

    return merge_right_cost, merge_left_cost

def merge_clusters(root, clusters, k):
    while len(clusters) > k:
        # Placeholder variables to find best merge location
        min_cost = np.inf
        min_dist = 0
        min_i = 0
        to_be_merged = None
        merge_receiver = None
        left = False

        for i in range(len(clusters) - 1):
            left = clusters[i].center
            right = clusters[i+1].center
            
            # Get cost of merging between left and right clusters
            parent_path = get_lca_path(left, right)
            dist = get_node(root, parent_path).dist
            merge_right_cost, merge_left_cost = merge_costs(dist, clusters[i], clusters[i+1])

            # Track all necessary optimal merge parameters
            if min(merge_right_cost, merge_left_cost) < min_cost:
                left_right = merge_right_cost < merge_left_cost
                min_i = i
                if not left_right:
                    left_right = -1 
                    min_i = i + 1
                to_be_merged = clusters[min_i]
                merge_receiver = clusters[min_i + left_right]
                min_cost = min(merge_right_cost, merge_left_cost)
                min_dist = dist

        # Merge the smaller cluster into the bigger cluster. Delete the smaller cluster.
        merge_receiver.peak = get_node(root, get_lca_path(merge_receiver.peak, to_be_merged.peak))
        merge_receiver.points += to_be_merged.points
        clusters.pop(min_i)

    return clusters
                
def cluster_tree(root, subroot, k):
    if len(subroot) <= k:
        clusters = []
        # If this is a single leaf
        if subroot.is_leaf:
            new_cluster = Cluster(subroot, [subroot], subroot)
            return [new_cluster]
        # Else this is a subtree and we want a cluster per leaf
        for leaf in subroot.children:
            point_cluster = Cluster(leaf, [leaf], leaf)
            clusters.append(point_cluster)

        return clusters

    clusters = []
    if subroot.has_left_tree:
        clusters += cluster_tree(root, subroot.left_tree, k)
    if subroot.has_right_tree:
        clusters += cluster_tree(root, subroot.right_tree, k)
    clusters = merge_clusters(root, clusters, k)
    return clusters

def deprune_cluster(node):
    """ Find the cluster's maximum parent and set all the children of that node to be in the cluster """
    if node.is_leaf:
        return [node.point_id]

    points = []
    if node.left_tree is not None:
        points += deprune_cluster(node.left_tree)
    if node.right_tree is not None:
        points += deprune_cluster(node.right_tree)

    return points

def finalize_clusters(clusters):
    """ We could have the setting where
          o
         / \
        o   o
       /     \
      X       X
     / \     / \
    o   o   o   o
    is the pruned set of clusters, where X marks the peak of each cluster.
    If that is the case, we actually want the following
          o
         / \
        X   X
       /     \
      o       o
     / \     / \
    o   o   o   o
    if the new X positions are still less than the maximum epsilon.
    """
    epsilons = np.array([c.peak.dist for c in clusters])
    nonzeros = epsilons[np.where(epsilons > 0)]
    if nonzeros.shape[0] == 0:
        max_eps = 1e-8
    else:
        max_eps = np.max(nonzeros) + 1e-8

    for cluster in clusters:
        while cluster.peak.parent.dist < max_eps:
            cluster.peak = cluster.peak.parent
    return clusters

def dc_kcenter(root, num_points, k, min_points, with_noise=True):
    # k-center has the option of accounting for noise points
    if with_noise:
        pruned_root = copy_tree(root, min_points)
    else:
        pruned_root = root

    clusters = cluster_tree(pruned_root, pruned_root, k=k)
    clusters = finalize_clusters(clusters)
    #print("clusters:", clusters)
    for cluster in clusters:
        cluster.points = deprune_cluster(cluster.peak.orig_node)

    return clusters

def get_cluster_metadata(clusters, num_points, k):
    # Nodes that were never put into a cluster will have pred_label -1 (noise points)
    pred_labels = -1 * np.ones(num_points)
    centers = np.zeros(k, dtype=np.int32)
    for i, cluster in enumerate(clusters):
        if cluster.center.orig_node is None:
            centers[i] = cluster.center.point_id
        else:
            centers[i] = cluster.center.orig_node.children[0].point_id
        for point in cluster.points:
            pred_labels[point] = i
    epsilons = np.array([c.peak.dist for c in clusters])

    return pred_labels, centers, epsilons

def dc_clustering(root, num_points, k=4, min_points=1, with_noise=True):
    clusters = dc_kcenter(root, num_points, k, min_points, with_noise=with_noise)

    return get_cluster_metadata(clusters, num_points, k)

