
class KMEANS(object):

  def __init__(self, *, eps, min_pts, dist_method='euclidean', cluster_type="standard"):
        if cluster_type not in ['standard', 'corepoints']:
            raise AssertionError("Please select 'standard' or 'corepoints' for the cluster_type parameter.")
        self.eps = eps
        self.min_pts = min_pts
        self.dist_method = dist_method
        self.labels_ = None
        self.type_ = cluster_type

  def naive_kmeans():


