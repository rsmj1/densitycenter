import efficientdcdist.dctree as dcdist
import numpy as np


class KMEANS(object):

  def __init__(self, *, k):
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.labels = None #the kmeans applied will put the cluster labels here

  def naive_dc_kmeans(self, points, minPts, max_iters=100):
    '''
      Naive dc Kmeans uses LLoyd's algorithm, in each iteration using the point closest to all others as its "mean" rather than computing the mean explicitly
      By default only takes as input a dc_dist tree, and is not intended to work with other distance functions. 
    '''
    dc_tree = dcdist.DCTree(points, min_points=minPts)
    n = points.shape[0]
    cluster_center_indexes = np.random.choice(n, self.k, False)
    cluster_assignments = None

    for i in range(max_iters):
      #Assign points to closest cluster
      dists = np.array([[dc_tree.dc_dist(c,p_index) for c in cluster_center_indexes] for p_index in range(n)])
      cluster_assignments = np.argmin(dists, axis=1)

      #No need to create new clusters for last iteration
      if i == max_iters -1:
         continue
      #Create new clusters by finding point with smallest distance to all other points within cluster
      #This is currently an n^2 operation...
      for c in range(self.k):
         cluster_indexes = np.nonzero(cluster_assignments == c)[0]
         mean_dists = [np.mean([dc_tree.dc_dist(point, other) for other in cluster_indexes]) for point in cluster_indexes]
         cluster_center_indexes[c] = cluster_indexes[np.argmin(mean_dists)]
        
    self.labels = cluster_assignments
    print("Final assignments:", self.labels)






#Basic testing:
#points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
points = np.array([[1,1],[2,2], [3,3], [3,2], [1,2], 
                   [15,15],[16,16], [17,17], [17,15], [17,16]])

kmeans = KMEANS(k=2)
kmeans.naive_dc_kmeans(points=points, minPts=2, max_iters=5)
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
