import efficientdcdist.dctree as dcdist
import numpy as np
import heapq

class DCKMedian(object):

  def __init__(self, *, k, min_pts):
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.min_pts = min_pts

        self.labels_ = None #the kmeans applied will put the cluster labels here
        self.centers = None #The point values of the centers
        self.center_indexes = None #The point indexes in the point list of the centers


  def fit(self, points):
    '''
    Solves the K-median problem optimally over the dc-distance. 
    Currently with O(n^2k^2) complexity.
    '''
    self.simple_greedy(points)


  def simple_greedy(self, points):
    '''
    Solves the K-median problem optimally over the dc-distance with O(n^2k^2) complexity.
    It does so by greedily choosing the next center as the point that reduces the cost the most. 
    '''
    n = points.shape[0]
    dc_tree = dcdist.DCTree(points, min_points=self.min_pts, n_jobs=1)
    centers = []
    centers_lookup = set()
    for i in range(self.k):
        centers.append(-1)
        best_value = np.inf
        best_new_point = -1
        for j in range(n):
           if j in centers_lookup:
              continue
           centers[i] = j
           curr_value = self.kmedian_loss(points, centers, dc_tree)
           #print(centers, "curr_value:", curr_value)
           if curr_value < best_value:
              best_value = curr_value
              best_new_point = j
        print("best value:", best_value)
        centers[i] = best_new_point
        centers_lookup.add(best_new_point)

    self.labels_ = self.assign_points(points, centers, dc_tree)
    self.center_indexes = centers
    self.centers = points[centers]

  def kmedian_loss(self, points, centers, dc_tree):
    '''
    Computes the K-means loss, given k provided centers: 
      Sum for each point in points: dist from point to closest center
    
    Parameters
    ----------

    points : Numpy.array
      The set of points over which the loss is computed.
    
    centers : Numpy.array
      The indexes into points of the chosen centers.
    
    dc_tree : DCTree
      The dc-tree over the points.
    '''     
    n = points.shape[0]
    dists = np.array([[dc_tree.dc_dist(c,p_index) for c in centers] for p_index in range(n)])
    cluster_dists = np.min(dists, axis=1)
    loss = np.sum(cluster_dists)
    
    return loss

 

  def assign_points(self, points, centers, dc_tree):
    n = points.shape[0]
    dists = np.array([[dc_tree.dc_dist(c,p_index) for c in centers] for p_index in range(n)])
    cluster_assignments = np.argmin(dists, axis=1)
    #print("cluster_assignments:", cluster_assignments)
    return cluster_assignments

  def assign_points_eff(self, points, centers, dc_tree):
    '''
    Assigns the points bottom up to the provided centers in O(n).
    It starts by annotating the tree with the centers.
    It then bottom up creates a list of tuples of the point and its center. 
    Finally, this list is rearranged into the standard format of labelling.
    It resolves distance ties to the leftmost center in the subtree as an arbitrary choice.
    '''
    n = points.shape[0]
    self.mark_center_paths(centers, dc_tree)

    tuple_list = self.label_points_tuples(points, dc_tree)

    labels = np.zeros(n)
    for tup in tuple_list:
       i = tup[0]
       label = tup[1]
       labels[i] = label
        
    return labels



  def label_points_tuples(self, points, dc_tree):
     

     return #List of tuples

  def efficient_greedy(self, points):
     '''
     Solves the K-median problem optimally with complexity O(n * log(n)).
     It does so by greedily choosing the next center as the point that reduces the cost the most. 
     
     This is the approach that sorts a list.

     The other approach would use heapq. 
     '''
     n = points.shape[0]
     dc_tree = dcdist.DCTree(points, min_points=self.min_pts, n_jobs=1)
     annotations = self.annotate_tree(dc_tree)

     annotations.sort(reverse=True, key=lambda x : x[0]) #Sort by the first value of the tuples - the potential cost-decrease. Reverse=True to get descending order.
     cluster_centers = set() # We should not use the "in" operation, as it is a worst-case O(n) operation. Just add again and again

     for annotation in annotations:
        curr_len = len(cluster_centers)
        if curr_len >= self.k:
           break
        cluster_centers.add(annotation[1])
        new_len = len(cluster_centers)
        if curr_len != new_len: #This janky setup is to make sure we do not need to use the "in" operation which has a worst-case O(n) complexity
           annotation[2].chosen = True
           annotation[2].best_center = annotation[1] 
     # Now we just need to assign the points to the clusters.

     return
  
  def annotate_tree(self, tree): #This is the cost_computation algorithm
     '''
     Does bottom-up on the dc-tree, generating an array of the annotations. We do not need to keep track of which tree-node generated which annotation
     , so the order in which they are inserted into the array does not matter. 
     Appending to a list is amortized O(n)
     '''

     output = []
     def annotation_builder(tree, array, parent_dist):
        if tree.is_leaf:
          array.append((parent_dist, tree.point_id, tree)) #Need to append a pointer to the tree itself because we want to mark the spots in the tree form which a point was chosen.
          return 0, tree.point_id, 1 #Returns local cost, the center for that cost and the size of the tree
        else:
           left_cost, left_center, left_size = annotation_builder(tree.left_tree, tree.dist)
           right_cost, right_center, right_size =  annotation_builder(tree.right_tree, tree.dist)
           total_size = left_size + right_size

           cost_left_center = left_cost + right_size * tree.dist
           cost_right_center = right_cost + left_size * tree.dist

          #We do not need to keep track of whether we are in the root.
           if cost_left_center > cost_right_center: #C_L > C_R implies higher decrease for C_R
              cost_decrease = parent_dist* total_size - cost_right_center 
              array.append((cost_decrease, right_center, total_size))
              return cost_right_center, right_center, total_size
           else:
              cost_decrease = parent_dist* total_size - cost_left_center 
              array.append((cost_decrease, left_center, total_size))
              return cost_left_center, left_center, total_size
     annotation_builder(tree, output, np.inf)
     print("output after the call:", output)
     return output
  
#Basic testing:
#points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
#points = np.array([[1,1],[2,2], [3,3], [3,2], [1,2], 
#                   [15,15],[16,16], [17,17], [17,15], [17,16]])

#kmeans = KMEANS(k=2)
#kmeans.plusplus_dc_kmeans(points=points, minPts=2, max_iters=5)
#kmeans.naive_dc_kmeans(points=points, minPts=2, max_iters=5)





'''
 Examples
    --------
    >>> import dcdist
    >>> points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
    >>> dc_tree = dcdist.DCTree(points, 5)
    >>> print(dc_tree.dc_dist(2,5))
    >>> print(dc_tree.dc_distances(range(len(points))))
    >>> print(dc_tree.dc_distances([0,1], [2,3]))

    >>> s = dcdist.serialize(dc_tree)
    >>> dc_tree_new = dcdist.deserialize(s)

    >>> b = dcdist.serialize_compressed(dc_tree)
    >>> dc_tree_new = dcdist.deserialize_compressed(b)

    >>> dcdist.save(dc_tree, "./data.dctree")
    >>> dc_tree_new = dcdist.load("./data.dctree")
'''
