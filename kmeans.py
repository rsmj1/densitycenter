import efficientdcdist.dctree as dcdist
import numpy as np


class KMEANS(object):

  def __init__(self, *, k):
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.labels = None #the kmeans applied will put the cluster labels here

  def basic_dc_lloyds(self, points, dc_tree, cluster_center_indexes, max_iters=100):
    '''
      Naive dc Kmeans uses LLoyd's algorithm, in each iteration using the point closest to all others as its "mean" rather than computing the mean explicitly
      By default only takes as input a dc_dist tree, and is not intended to work with other distance functions. 
    '''
    n = points.shape[0]
    cluster_assignments = None

    for i in range(max_iters):
      old_center_indexes = cluster_center_indexes.copy() #Save the old indexes for termination criteria
      #Assign points to closest cluster
      dists = np.array([[dc_tree.dc_dist(c,p_index) for c in cluster_center_indexes] for p_index in range(n)])
      cluster_assignments = np.argmin(dists, axis=1)

      #No need to create new clusters for last iteration
      if i == max_iters -1:
         print("Max iters reached")
         continue
      
      #Create new clusters by finding point with smallest distance to all other points within cluster
      #This is currently an n^2 operation...
      #TODO: This can be improved by using the distances method which will make the matrix very efficiently. From there, we just take the mean in each row.
      #However, still n^2 to take the mean in each row
      for c in range(self.k):
         cluster_indexes = np.nonzero(cluster_assignments == c)[0]
         mean_dists = [np.mean([dc_tree.dc_dist(point, other) for other in cluster_indexes]) for point in cluster_indexes]
         cluster_center_indexes[c] = cluster_indexes[np.argmin(mean_dists)]

      if np.array_equal(cluster_center_indexes, old_center_indexes):
        print("Stable point reached, stopping at iteration", i)
        break
  
    self.labels = cluster_assignments
    #print("Final assignments:", self.labels)


  def plusplus_dc_kmeans(self, points, minPts, max_iters=100):
    n = points.shape[0]
    dc_tree = dcdist.DCTree(points, min_points=minPts)

    '''
    Choose one center uniformly at random among the data points.
    For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
    Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
    Repeat Steps 2 and 3 until k centers have been chosen.
    Now that the initial centers have been chosen, proceed using standard k-means clustering.
    '''
    #Keep set of points that might be chosen as new centers
    #I use a numpy array rather than a set since choosing a random element from a set is inefficient and furthermore does not allow me to use the dc_tree.distances() function
    if self.k >= n:
       raise AssertionError("k should be smaller than the number of points...")
    
    point_indexes = np.arange(n)
    cluster_center_indexes = np.zeros(self.k, dtype=np.int64)
    #Choose first center uniformly at random
    first_choice = np.random.choice(n, 1)
    cluster_center_indexes[0] = first_choice

    #Remove first center from pool of possible points to choose. If not, cycles of chosen points might occur. 
    point_indexes = point_indexes[point_indexes!= first_choice]

    #print("Distance 0, choice", dc_tree.dc_dist(0,cluster_center_indexes[0]), "Distance n, choice", dc_tree.dc_dist(n-1, cluster_center_indexes[0]))
    for i in range(1, self.k):
       #Compute the minimal distance for each point that has yet to be chosen between it and all previously chosen centers
       #dists_to_cluster = dc_tree.dc_distances(np.array([cluster_center_indexes[i-1]]), point_indexes)
       dists_to_cluster = np.array([np.min([dc_tree.dc_dist(c,p) for c in cluster_center_indexes[:i]]) for p in point_indexes])
       #Choose next point according to pdf based on D(x')^2/sum_x\inX(D(x)^2). Squeeze ensures proper shape of array, from (1,n) to (n,)
       squared_dists = dists_to_cluster ** 2
       pdf = np.squeeze(squared_dists / squared_dists.sum())
       choice = np.random.choice(point_indexes, 1, p=pdf)

       #Remove next chosen point from pool of possible points to choose. If not, cycles of chosen points might occur. 
       point_indexes = point_indexes[point_indexes!= choice]

       cluster_center_indexes[i] = choice

       
    #print("final indexes:", cluster_center_indexes)
    self.basic_dc_lloyds(points, dc_tree, cluster_center_indexes, max_iters)
  
  def naive_dc_kmeans(self, points, minPts, max_iters=100):
    n = points.shape[0]
    dc_tree = dcdist.DCTree(points, min_points=minPts)
    cluster_center_indexes = np.random.choice(n, self.k, False)
    self.basic_dc_lloyds(points, dc_tree, cluster_center_indexes, max_iters)



#Basic testing:
#points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
points = np.array([[1,1],[2,2], [3,3], [3,2], [1,2], 
                   [15,15],[16,16], [17,17], [17,15], [17,16]])

kmeans = KMEANS(k=2)
kmeans.plusplus_dc_kmeans(points=points, minPts=2, max_iters=5)
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
