import efficientdcdist.dctree as dcdist
import numpy as np


class KMEANS(object):

  def __init__(self, *, k):
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.labels = None #the kmeans applied will put the cluster labels here

  def naive_dc_kmeans(self, points, minPts):
    '''
      Naive dc Kmeans uses LLoyd's algorithm, in each iteration using the point closest to all others as its "mean" rather than computing the mean explicitly
      By default only takes as input a dc_dist tree, and is not intended to work with other distance functions. 
    '''
    dc_tree = dcdist.DCTree(points, min_points=minPts)





#Basic testing:
points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])

kmeans = KMEANS(k=2)
kmeans.naive_dc_kmeans(points=points, minPts=3)
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
