import numpy as np

def create_hierarchical_clusters(n, point=None, std_dev=100, d=2, recur_likelihood=1.0, unique_vals = False):
    """
    Create a dataset that samples from a Gaussian distribution.
    For each sample, recursively sample points from its neighborhood with some likelihood.
    In expectation, this creates a normally distributed ball of Gaussian clusters
        such that each cluster likely has sub-clusters within it.
    """
    if point is None:
        point = np.zeros(d)
    if n <= 1:
        return np.random.multivariate_normal(point, np.eye(d) * std_dev, 1)
    
    points = []
    points_remaining = n
    i = 0
    while points_remaining > 1:
        subcluster_size = int(np.random.uniform() * points_remaining)
        if np.random.uniform() < recur_likelihood:
            subcluster_mean = np.random.multivariate_normal(point, np.eye(d) * std_dev)
            subcluster = create_hierarchical_clusters(
                n=subcluster_size,
                point=subcluster_mean,
                std_dev=std_dev/10,
                d=d,
                recur_likelihood=recur_likelihood * np.random.uniform()
            )
        else:
            subcluster = np.random.multivariate_normal(point, np.eye(d) * std_dev, subcluster_size)
        points.append(subcluster)
        points_remaining -= subcluster_size
            
    if points:
        points = np.concatenate(points, axis=0)
        if len(points) > n:
            points = points[np.random.choice(len(points), n)]
        if unique_vals:
            print("Only returning unique values, might have less than", n, "samples")
            return np.unique(points, axis=0)
        return points
    return np.empty([0, d])