import efficientdcdist.dctree as dcdist
import numpy as np
from density_tree import make_tree
from cluster_tree import prune_tree

class DCKMeans(object):

  def __init__(self, *, k, min_pts, method="optimal", max_iters=100, noise_mode="none"):
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.min_pts = min_pts
        self.method = method # The methods are "naive", "plusplus", "hungry"
        self.max_iters = max_iters # How many iterations of Lloyd's algorithm to maximally run 

        self.labels_ = None #the kmeans applied will put the cluster labels here
        self.centers = None #The point values of the centers
        self.center_indexes = None #The point indexes in the point list of the centers
        self.noise_mode = noise_mode #The execution mode - either no noise detection, "medium" noise detection or full noise detection


  def fit(self, points):
    if self.method == "plusplus":
      #print("doing plusplus")
      #print("labels are now:", self.labels_)
      self.plusplus_dc_kmeans(points, self.min_pts)
      #print("labels after:", self.labels_)
    elif self.method == "naive":
      self.naive_dc_kmeans(points, self.min_pts)
    elif self.method == "optimal":
       #self.solve(points, self.min_pts)
       self.efficient_greedy(points)



  def solve(self, points):
      '''
      Solves the K-means problem optimally over the dc-distance in O(n^2k time).
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
            curr_value = self.kmeans_loss(points, centers, dc_tree)
            #print(centers, "curr_value:", curr_value)
            if curr_value < best_value:
                best_value = curr_value
                best_new_point = j
          #print("best point:", best_new_point)
          centers[i] = best_new_point
          centers_lookup.add(best_new_point)

      self.labels_ = self.assign_points(points, centers, dc_tree)
      self.center_indexes = centers
      self.centers = points[centers]


  def basic_dc_lloyds(self, points, dc_tree, cluster_center_indexes, max_iters=100):
    '''
      Naive dc Kmeans uses LLoyd's algorithm, in each iteration using the point closest to all others as its "mean" rather than computing the mean explicitly
      By default only takes as input a dc_dist tree, and is not intended to work with other distance functions. 
    '''
    n = points.shape[0]
    cluster_assignments = None

    for i in range(max_iters):
      old_center_indexes = cluster_center_indexes.copy() #Save the old indexes for termination criteria
      #print("Start of iteration cluster indexes:", old_center_indexes)
      #Assign points to closest cluster
      dists = np.array([[dc_tree.dc_dist(c,p_index) for c in cluster_center_indexes] for p_index in range(n)])
      #print("Distances to these clusters:", dists)
      cluster_assignments = np.argmin(dists, axis=1)
      #print("Yielding the following cluster assignments:", cluster_assignments)
      #No need to create new clusters for last iteration
      if i == max_iters -1:
         print("Max iters reached") 
         continue
      
      #Create new clusters by finding point with smallest distance to all other points within cluster
      #This is currently an n^2 operation...
      for c in range(self.k):
         #Find indexes of points part of cluster c
         cluster_indexes = np.nonzero(cluster_assignments == c)[0]
         #print("Indexes of cluster" + str(c)+" are:", cluster_indexes)
         mean_dists = [np.mean([dc_tree.dc_dist(point, other) for other in cluster_indexes]) for point in cluster_indexes]
         #print("mean_dists:", mean_dists)
         cluster_center_indexes[c] = cluster_indexes[np.argmin(mean_dists)]
         #print("chose index:", cluster_center_indexes[c])

      if np.array_equal(cluster_center_indexes, old_center_indexes):
        #print("K is",self.k , "and stable point reached, stopping at iteration", i)
        #print("Cluster center indexes:", cluster_center_indexes)
        break
    
    self.center_indexes = cluster_center_indexes
    self.centers = points[cluster_center_indexes]
    self.labels_ = cluster_assignments
    #print("Final assignments:", self.labels)


  def plusplus_dc_kmeans(self, points, minPts, max_iters=100):
    '''
    Choose one center uniformly at random among the data points. \n
    For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen. \n
    Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2. \n
    Repeat Steps 2 and 3 until k centers have been chosen. \n
    Now that the initial centers have been chosen, proceed using standard k-means clustering. \n
    '''
    
    n = points.shape[0]
    dc_tree = dcdist.DCTree(points, min_points=minPts, n_jobs=1)
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
    dc_tree = dcdist.DCTree(points, min_points=minPts, n_jobs=1)
    cluster_center_indexes = np.random.choice(n, self.k, False)
    self.basic_dc_lloyds(points, dc_tree, cluster_center_indexes, max_iters)


  def kmeans_loss(self, points, centers, dc_tree):
    '''
    Computes the K-means loss, given k provided centers: Sum for each point in points: dist from point to closest center squared.
    
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
    loss = np.sum(cluster_dists**2)
    return loss

  def hungry_kmeans_loss(self, points, cluster_center_indexes, dc_tree):
    '''
    Computes the Hungry K-means loss, given k provided centers: 
      Sum for each center of number of elements in the cluster times the largest distance from the center to a point in the cluster.
    
    Parameters
    ----------

    points : Numpy.array
      The set of points over which the loss is computed.
    
    cluster_center_indexes : Numpy.array
      The indexes into points of the chosen centers.
    
    dc_tree : DCTree
      The dc-tree over the points.
    '''     
    n = points.shape[0]
    k = cluster_center_indexes.shape[0]
    #Compute the clusterings
    dists = np.array([[dc_tree.dc_dist(c,p_index) for c in cluster_center_indexes] for p_index in range(n)])
    cluster_assignments = np.argmin(dists, axis=1)

    #Find the max value rho for each cluster
    maxes = np.zeros(k)
    for i in range(k):
       maxes[i] = np.max(dists[cluster_assignments == i])
    #Find the number of objects in each cluster
    _, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
    #Compute the final loss
    loss = np.sum([max_dist * size for max_dist, size in zip(maxes, cluster_sizes)])
    return loss


  def assign_points(self, points, centers, dc_tree):
    n = points.shape[0]
    dists = np.array([[dc_tree.dc_dist(c,p_index) for c in centers] for p_index in range(n)])
    cluster_assignments = np.argmin(dists, axis=1)
    return cluster_assignments
  

  def assign_points_eff(self, points, dc_tree):
    '''
    Assigns the points bottom up to the provided centers in O(n).
    It bottom up creates a list of tuples of the point and its center. 
    Finally, this list is rearranged into the standard format of labelling.
    It resolves distance ties to the leftmost center in the subtree as an arbitrary choice.
    '''
    n = points.shape[0]

    tuple_list = self.label_points_tuples(dc_tree)

    labels = np.zeros(n)
    for tup in tuple_list:
       i = tup[0]
       label = tup[1]
       labels[i] = label
    labels = normalize_cluster_ordering(labels) #THIS STEP IS NOT O(n) DUE TO USING SET IN 
    return labels



  def label_points_tuples(self, dc_tree):
     '''
     This labels the points and outputs a list of tuples of the point_id and the center it is assigned to. 
     It is done this way to avoid having to use a global list. 
     '''
     output = []

     def list_builder(dc_tree, list, center):
        '''
        Helper method that does the recursive list building. 
        It remembers the last center seen on the path from the root to the leaf, which is the center that gets the point assigned. 
        '''
        if dc_tree.chosen:
              center = dc_tree.best_center

        if dc_tree.is_leaf:
           list.append((dc_tree.point_id, center))
        else:
           list_builder(dc_tree.left_tree, list, center)
           list_builder(dc_tree.right_tree, list, center)

     list_builder(dc_tree, output, -1)
     return output #List of tuples

  def efficient_greedy(self, points):
     '''
     Solves the K-median problem optimally with complexity O(n * log(n)).
     It does so by greedily choosing the next center as the point that reduces the cost the most. 
     
     This is the approach that sorts a list.

     The other approach would use heapq. Issue with this is that we need to re-annotate the tree anyways with the specific choices made in the end (or just set a marker to true).
     '''
     n = points.shape[0]
     placeholder = np.zeros(n)
     dc_tree, _ = make_tree(points, placeholder, min_points=self.min_pts, )
     annotations = self.annotate_tree(dc_tree)

    
     if self.noise_mode == "medium" or self.noise_mode == "full":
         #This is to avoid picking noise points as centers
         annotations = self.prune_annotations(annotations)

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

     #Now we just need to assign the points to the clusters.
     centers = list(cluster_centers)
     if self.noise_mode == "none":
        self.labels_ = self.assign_points_eff(points, dc_tree)
     elif self.noise_mode == "medium":
        self.mark_paths(dc_tree, centers)
        self.labels_ = self.assign_points_prune(points, dc_tree)
     elif self.noise_mode == "full":
        self.mark_paths(dc_tree, centers)
        self.labels_ = self.assign_points_prune_full(points, dc_tree)        
     else: 
        raise AssertionError("The noise detection mode is not recognized. Choose between none, medium or full.")
     
     print("labels:", self.labels_)
     self.center_indexes = centers
     self.centers = points[centers]
     return
  
  def annotate_tree(self, tree):
     '''
     Does bottom-up on the dc-tree, generating an array of the annotations. We do not need to keep track of which tree-node generated which annotation
     , so the order in which they are inserted into the array does not matter. 
     Appending to a list is amortized O(n).
     It follows the cost_computation algorithm from my paper.

     Since this is for K-means, notice that distances are squared whenever used!!
     '''

     output = []
     def annotation_builder(tree, array, parent_dist):
        if tree.is_leaf:
          array.append((parent_dist**2, tree.point_id, tree)) #Need to append a pointer to the tree itself because we want to mark the spots in the tree form which a point was chosen.
          #print("Annotation:", parent_dist, tree.point_id,  self.get_leaves(tree))
          return 0, tree.point_id, 1 #Returns local cost, the center for that cost and the size of the tree
        else:
           left_cost, left_center, left_size = annotation_builder(tree.left_tree, array, tree.dist)
           right_cost, right_center, right_size =  annotation_builder(tree.right_tree, array, tree.dist)
           total_size = left_size + right_size

           cost_left_center = left_cost + right_size * (tree.dist**2)
           cost_right_center = right_cost + left_size * (tree.dist**2)

          #We do not need to keep track of whether we are in the root.
           if cost_left_center > cost_right_center: #C_L > C_R implies higher decrease for C_R
              cost_decrease = (parent_dist**2) * total_size - cost_right_center 
              array.append((cost_decrease, right_center, tree))
              #print("Annotation:", cost_decrease, right_center, self.get_leaves(tree))

              return cost_right_center, right_center, total_size
           else:
              cost_decrease = (parent_dist**2) * total_size - cost_left_center 
              array.append((cost_decrease, left_center, tree))
              #print("Annotation:", cost_decrease, left_center,  self.get_leaves(tree))

              return cost_left_center, left_center, total_size
     annotation_builder(tree, output, np.inf)
     return output
  
  def get_leaves(self, dc_tree):
        '''
        Returns the set of ids of the leaf nodes within the given cluster.
        '''
        if dc_tree.is_leaf:
            return [dc_tree.point_id]
        else:
            #print("left:", self.get_leaves(dc_tree.left_tree))
            #print("right:", self.get_tree_size(dc_tree.right_tree))
            return self.get_leaves(dc_tree.left_tree) + self.get_leaves(dc_tree.right_tree)
  
  def assign_points_prune(self, points, dc_tree):
     '''
     This method assigns all points that have the same distance to multiple centers to noise.

     Thought TODO: If we prune the tree and smooth out (remove internal nodes where no left/right child is left), then feasibly one could run K-median on it. 
     This might be equivalent to the pruning of the output list of annotations(?)
     '''
     output = []

     def list_builder(dc_tree, list):
        '''
        Inefficient implementation for now - proof of concept.
        Helper method that does the recursive list building.
        Returns a list of tuples with the points and the center they are assigned to. 
        '''
        if dc_tree.is_leaf:
            if dc_tree.center_path:
              print("here2")

              list.append((dc_tree.point_id, dc_tree.unique_center))
              return []
            else:
               return [dc_tree.point_id]
        else:
            left_not_assigned = list_builder(dc_tree.left_tree, list)
            right_not_assigned = list_builder(dc_tree.right_tree, list)

            all_not_assigned = left_not_assigned + right_not_assigned
            if dc_tree.center_path: #On center path they should be assigned
               print("here1")
               if dc_tree.num_centers == 1:
                  for p in all_not_assigned:
                     list.append((p, dc_tree.unique_center))
               else: #Not uniquely close to one center
                  for p in all_not_assigned:
                     list.append((p, -1))
               return []
            else: #Not on center path - cannnot be assigned yet
               return all_not_assigned


     list_builder(dc_tree, output)
     print("output:", output)
     n = points.shape[0]

     labels = np.zeros(n)
     for tup in output:
      i = tup[0]
      label = tup[1]
      labels[i] = label
     labels = normalize_cluster_ordering(labels) #THIS STEP IS NOT O(n) DUE TO USING SET IN 
     print("labels:", labels)
     return labels
        


  def assign_points_prune_full(self, points, dc_tree):
     '''
     This method assigns all points that have the same distance to multiple centers to noise.

     Thought TODO: If we prune the tree and smooth out (remove internal nodes where no left/right child is left), then feasibly one could run K-median on it. 
     This might be equivalent to the pruning of the output list of annotations(?)
     '''
     output = []

     def list_builder(dc_tree, list):
        '''
        Inefficient implementation for now - proof of concept.
        Helper method that does the recursive list building.
        Returns 0 from a single-center path and 1 from an "unknown" path, and 2 from a multi-center path.
        '''
        if dc_tree.is_leaf:
            if dc_tree.center_path:
              return 0
            else:
               return 1
        else:
            left_path= list_builder(dc_tree.left_tree, list)
            right_path = list_builder(dc_tree.right_tree, list)

            if dc_tree.center_path: #On center path they should be assigned

               if dc_tree.num_centers == 1:
                  return 0
               else: #Not uniquely close to one center
                  #Now, those points that come from a single-center path should now be pruned together via external method.
                  #Those that don't should be assigned to noise
                  points, noise = [],[]
                  if left_path == 0:
                     points, noise = self.prune_cluster_subtree(dc_tree.left_tree, self.min_pts)
                     center = dc_tree.left_tree.unique_center
                     for point in points:
                        list.append((point, center))  
                     for point in noise:
                        list.append((point, -1))
                  elif left_path == 1:

                     noise = self.get_leaves(dc_tree.left_tree)
                     for point in noise:
                        list.append((point, -1))
                     #Assign to noise

                  if right_path == 0:
                     points, noise = self.prune_cluster_subtree(dc_tree.right_tree, self.min_pts)
                     center = dc_tree.right_tree.unique_center
                     for point in points:
                        list.append((point, center)) 
                     for point in noise:
                        list.append((point, -1))
                  elif right_path == 1:
                     noise = self.get_leaves(dc_tree.right_tree)
                     for point in noise:
                        list.append((point, -1))    
                  
                  return 2 #Return false here - 
            
            else: #Not on center path - cannnot be assigned yet
               return 1

     list_builder(dc_tree, output)
     n = points.shape[0]

     labels = np.zeros(n)
     for tup in output:
      i = tup[0]
      label = tup[1]
      labels[i] = label
     labels = normalize_cluster_ordering(labels) #THIS STEP IS NOT O(n) DUE TO USING SET IN 
     return labels


  def prune_cluster_subtree(self, dc_tree, min_pts):
   '''
   Will prune the tree and return the points that should be pruned and those that should be assigned.
   '''
   pruned_tree = prune_tree(dc_tree, min_pts)

   def noise_collector(tree, points_list, noise_list):
      if tree.is_leaf:
         if tree.point_id < 0: #This is noise
            noise_list += self.get_leaves(tree.orig_node)
            return
         else:
            points_list += [tree.point_id]
      else:
         l_point_id = tree.left_tree.point_id
         r_point_id = tree.right_tree.point_id
         if l_point_id is None and r_point_id is None: #If these are none they are internal nodes - ergo we have a split in the pruned tree.
            points_list += self.get_leaves(tree.orig_node)
            return
         
         noise_collector(tree.left_tree, points_list, noise_list)
         noise_collector(tree.right_tree, points_list, noise_list)

   points, noise = [],[]

   if pruned_tree is None: #If everything is pruned
      noise += self.get_leaves(dc_tree)
   else:
      noise_collector(pruned_tree, points, noise)

   return points, noise




  def mark_paths(self, dc_tree, centers):
     '''
     Inefficient implementation - it is n^2
     This marks in the tree the paths from the centers and how many centers are on overlapping paths.
     '''
     if dc_tree.is_leaf:
        if dc_tree.point_id in centers:
           dc_tree.center_path = True
           dc_tree.num_centers = 1
           dc_tree.unique_center = dc_tree.point_id
           return [dc_tree.point_id]
        return []
     else:
        left_path = self.mark_paths(dc_tree.left_tree, centers)
        right_path = self.mark_paths(dc_tree.right_tree, centers)
        union_path = left_path + right_path
        num_centers = len(union_path)
        if num_centers > 0:
           dc_tree.center_path = True
           dc_tree.num_centers = num_centers
           dc_tree.unique_center = union_path[0]
        return union_path
     


def normalize_cluster_ordering(cluster_labels):
    '''
    Normalizes the clustering labels so that the first cluster label encountered is labelled 0, the next 1 and so on. 
    Preserves noise labelled as -1. Useful for clustering comparisons.
    '''
    #print("cluster_labels before:", cluster_labels)
    n = cluster_labels.shape[0]
    cluster_index = 0
    cluster_index_mapping = {}
    mapped_labels = set()
    mapped_labels.add(-1)
    norm_cluster_labels = np.empty(n, dtype=np.int64)

    for i, label in enumerate(cluster_labels):
        if label not in mapped_labels:
            #Create mapping for new encountered cluster
            mapped_labels.add(label)
            cluster_index_mapping[label] = cluster_index
            cluster_index += 1
        
        if label == -1:
            #Preserve noise labellings
            norm_cluster_labels[i] = -1
        else:
            norm_cluster_labels[i] = cluster_index_mapping[label]

    #print("cluster_labels after:", norm_cluster_labels)
    return norm_cluster_labels
