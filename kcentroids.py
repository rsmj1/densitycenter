import efficientdcdist.dctree as dcdist
import numpy as np
import heapq
import numba
from density_tree import make_tree
from cluster_tree import prune_tree, copy_tree
class DCKCentroids(object):

  def __init__(self, *, k, min_pts, loss="kmedian", noise_mode="none"):
        '''
        This will perform either K-means, K-median (or in the future also other similar loss approaches) over the binary dc-tree.

        Parameters
        ----------
        k : int
            The number of centers to find
        min_pts : int
            The number of points for something to be a core-point.
        loss : String, default: "kmedian"
            The loss function and subsequently general K-method. Options: "kmeans", "kmedian",
        noise_mode : String, default: "none"
            The way it should handle noise, either ignoring it or detecting various levels of noise. Options: "none", "medium", "full".

        '''
        self.k = k # The number of clusters that the given kmeans algorithm chosen will use
        self.min_pts = min_pts
        self.loss = loss
        self.labels_ = None #the kmeans applied will put the cluster labels here
        self.centers = None #The point values of the centers
        self.center_indexes = None #The point indexes in the point list of the centers
        self.noise_mode = noise_mode #The execution mode - either no noise detection, "medium" noise detection or full noise detection

        self.cdists = None


  def fit(self, points):
    '''
    Solves the K-median / K-means problem optimally over the dc-distance and (binary) dc-tree. 
    '''
    print("Running " + self.loss + " with noise detection: " + self.noise_mode)
    self.cdists = self.get_cdists(points, self.min_pts)
    self.efficient_greedy(points)
    #self.simple_greedy(points)

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

      annotations = self.annotate_tree(dc_tree) #Will use K-means or K-median loss depending on self.loss


      if self.noise_mode == "medium" or self.noise_mode == "full":
            #This is to avoid picking noise points as centers
            #annotations1 = self.prune_annotations_other(dc_tree, annotations) #This prunes based on the pruned tree. This is always slightly less aggressive than the one below. 
            annotations = self.prune_annotations(annotations) #This prunes based on occurences in the annotation list.

      annotations.sort(reverse=True, key=lambda x : x[0]) #Sort by the first value of the tuples - the potential cost-decrease. Reverse=True to get descending order.
      cluster_centers = set() #We should not use the "in" operation, as it is a worst-case O(n) operation. Just add again and again

      # print("Annotations BINARY:")
      # display_annos = [(x[0], x[1], self.get_leaves(x[2])) for x in annotations]
      # print(display_annos)

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
         self.mark_center_paths(dc_tree, centers)
         self.labels_ = self.assign_points_prune(points, dc_tree)
      elif self.noise_mode == "full":
         self.mark_center_paths(dc_tree, centers)
         #self.labels_ = self.assign_points_prune_full(points, dc_tree) 
         self.labels_ = self.assign_points_prune_stability(dc_tree, self.cluster_stability_experimental)      
         
      else: 
         raise AssertionError("The noise detection mode is not recognized. Choose between none, medium or full.")
      
      print("labels BINIARY:", self.labels_)

      self.center_indexes = centers
      self.centers = points[centers]
      return

  def simple_greedy(self, points):
    '''
    Solves the K-median / K-means problem optimally over the dc-distance with O(n^2k^2) complexity.
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
           if self.loss == "kmedian":
            curr_value = self.kmedian_loss(points, centers, dc_tree)
           elif self.loss == "kmeans":
            curr_value = self.kmeans_loss(points, centers, dc_tree)
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


  def assign_points(self, points, centers, dc_tree):
    n = points.shape[0]
    dists = np.array([[dc_tree.dc_dist(c,p_index) for c in centers] for p_index in range(n)])
    cluster_assignments = np.argmin(dists, axis=1)
    #print("cluster_assignments:", cluster_assignments)
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
    labels = self.normalize_cluster_ordering(labels) #THIS STEP IS NOT O(n) DUE TO USING SET IN 
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

  def assign_points_prune(self, points, dc_tree):
     '''
     This method assigns all points that have the same distance to multiple centers to noise.
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
     return self.tuple_labels_to_labels(output)
        

  def assign_points_prune_full(self, points, dc_tree):
     '''
     This method assigns all points that have the same distance to multiple centers to noise.
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
                  if left_path == 0: #Assign via pruned tree
                     points, noise = self.prune_cluster_subtree(dc_tree.left_tree, self.min_pts)
                     center = dc_tree.left_tree.unique_center
                     for point in points:
                        list.append((point, center))  
                     for point in noise:
                        list.append((point, -1))
                  elif left_path == 1: #Assign to noise
                     noise = self.get_leaves(dc_tree.left_tree)
                     for point in noise: 
                        list.append((point, -1))

                  if right_path == 0: #Assign via pruned tree
                     points, noise = self.prune_cluster_subtree(dc_tree.right_tree, self.min_pts)
                     center = dc_tree.right_tree.unique_center
                     for point in points:
                        list.append((point, center)) 
                     for point in noise:
                        list.append((point, -1))
                  elif right_path == 1: #Assign to noise
                     noise = self.get_leaves(dc_tree.right_tree)
                     for point in noise:
                        list.append((point, -1))    
                  
                  return 2 #Points from here are already assigned, so return 2 
            
            else: #Not on center path - cannnot be assigned yet
               return 1

     list_builder(dc_tree, output)
     print("output:", output)
     return self.tuple_labels_to_labels(output)

  def assign_points_prune_stability(self, dc_tree, stability_function):
     '''
     Assigns points to the centers based on stability computation. 
     '''
     output = []

     def list_builder(dc_tree, list, stability):
        '''
        Inefficient implementation for now - proof of concept.
        Helper method that does the recursive list building.
        Returns 0 from a single-center path and 1 from an "unknown" path, and 2 from a multi-center path.
        '''
        if dc_tree.is_leaf:
            if dc_tree.center_path:
              sta = stability(dc_tree, self.cdists)
              #print("nodes:", np.array(self.get_leaves(dc_tree))+1)
              #print("stability:", sta)
              return 0, stability(dc_tree, self.cdists), dc_tree
            else:
              return 1, 0, None #Not a center path - so no need to compute a stability and return any pointer to a tree.
        else:
            left_path, left_best_stability, left_best_cluster = list_builder(dc_tree.left_tree, list, stability)
            right_path, right_best_stability, right_best_cluster = list_builder(dc_tree.right_tree, list, stability)
            if dc_tree.center_path: #On center path they should be assigned

               if dc_tree.num_centers == 1: #Compute new stability since we are on a center path and return the best of the stabilities.
                  new_stability = stability(dc_tree, self.cdists)
                  sta = stability(dc_tree, self.cdists)
                  # print("nodes:", np.array(self.get_leaves(dc_tree))+1)
                  # print("stability:", sta)
                  
                  if left_path == 0: #Check if the center path comes from left or right and compare stabilities, choose the best one.
                     if new_stability >= left_best_stability:
                        return 0, new_stability, dc_tree
                     else:
                        return 0, left_best_stability, left_best_cluster
                  else:
                     if new_stability >= right_best_stability:
                        return 0, new_stability, dc_tree
                     else:
                        return 0, right_best_stability, right_best_cluster
               else: #Not uniquely close to one center (potentially anymore, we might be coming from a single center path)
                  points, noise = [],[]
                  if left_path == 0: #Assign via pruned tree
                     points = self.get_leaves(left_best_cluster)
                     print("points to assign l:", points)
                     center = dc_tree.left_tree.unique_center
                     noise = [p for p in self.get_leaves(dc_tree.left_tree) if p not in points]
                     print("points for noise l:", noise)
                     for point in points:
                        list.append((point, center))  
                     for point in noise:
                        list.append((point, -1))
                  elif left_path == 1: #Assign to noise
                     noise = self.get_leaves(dc_tree.left_tree)
                     for point in noise: 
                        list.append((point, -1))

                  if right_path == 0: #Assign via pruned tree
                     points = self.get_leaves(right_best_cluster)
                     print("points to assign r:", points)
                     center = dc_tree.right_tree.unique_center
                     noise = [p for p in self.get_leaves(dc_tree.right_tree) if p not in points]
                     print("points for noise r:", noise)                     
                     for point in points:
                        list.append((point, center)) 
                     for point in noise:
                        list.append((point, -1))
                  elif right_path == 1: #Assign to noise
                     noise = self.get_leaves(dc_tree.right_tree)
                     for point in noise:
                        list.append((point, -1))    
                  
                  return 2, 0, None #Points from here are already assigned, so return 2 
            
            else: #Not on center path - cannnot be assigned yet
               return 1, 0, None

     list_builder(dc_tree, output, stability_function)
     print("output:", output)
     return self.tuple_labels_to_labels(output)



  def tuple_labels_to_labels(self, tuples):
     '''
     Takes a list of tuples [(point, label),...,] and makes it into a standard cluster labelling. 
     '''
     
     n = len(tuples)
     labels = np.zeros(n)
     for tup in tuples:
      i = tup[0]
      label = tup[1]
      labels[i] = label
     labels = self.normalize_cluster_ordering(labels) #THIS STEP IS NOT O(n) DUE TO USING SET IN 
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
         if l_point_id is None and r_point_id is None: #If these ids are none they are internal nodes - ergo we have a split in the pruned tree.
            points_list += self.get_leaves(tree.orig_node)
            return
         
         noise_collector(tree.left_tree, points_list, noise_list)
         noise_collector(tree.right_tree, points_list, noise_list)

   points, noise = [],[]
   if pruned_tree is None: #If everything is pruned - then it should return the nodes - it does not make sense that everything is noise.. TODO
      points += self.get_leaves(dc_tree)
   else:
      noise_collector(pruned_tree, points, noise)

   return points, noise

  def prune_cluster_subtree_aggressive(self, dc_tree, min_pts):
   '''
   Will prune the tree and return the points that should be pruned and those that should be assigned.
   This keeps going to the leaves of the pruned tree no matter whether there is a split in the pruned tree or not. 
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
         noise_collector(tree.left_tree, points_list, noise_list)
         noise_collector(tree.right_tree, points_list, noise_list)

   points, noise = [],[]
   if pruned_tree is None: #If everything is pruned
      noise += self.get_leaves(dc_tree)
   else:
      noise_collector(pruned_tree, points, noise)

   return points, noise

  def mark_center_paths(self, dc_tree, centers):
     '''
     Inefficient implementation - it is n^2
     This marks in the tree the paths from the centers and how many centers are on overlapping paths.
     This is used for detecting when points are equidistant do multiple centers. 
     '''
     if dc_tree.is_leaf:
        if dc_tree.point_id in centers:
           dc_tree.center_path = True
           dc_tree.num_centers = 1
           dc_tree.unique_center = dc_tree.point_id
           return [dc_tree.point_id]
        return []
     else:
        left_path = self.mark_center_paths(dc_tree.left_tree, centers)
        right_path = self.mark_center_paths(dc_tree.right_tree, centers)
        union_path = left_path + right_path
        num_centers = len(union_path)
        if num_centers > 0:
           dc_tree.center_path = True
           dc_tree.num_centers = num_centers
           dc_tree.unique_center = union_path[0]
        return union_path

    ######################## Methods that are K-means / K-median specific ########################

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

    ######################## [END] Methods that are K-means / K-median specific ########################


  def annotate_tree(self, tree):
    '''
    Returns the list of annotations.
    '''
    output = []
    def annotation_builder(tree, array, parent_dist, loss):
        if tree.is_leaf:
            array.append((loss(parent_dist), tree.point_id, tree)) #Need to append a pointer to the tree itself because we want to mark the spots in the tree form which a point was chosen.
            #print("Annotation:", parent_dist, tree.point_id,  self.get_leaves(tree))
            return 0, tree.point_id, 1 #Returns local cost, the center for that cost and the size of the tree
        else:
            left_cost, left_center, left_size = annotation_builder(tree.left_tree, array, tree.dist, loss)
            right_cost, right_center, right_size =  annotation_builder(tree.right_tree, array, tree.dist, loss)
            total_size = left_size + right_size

            cost_left_center = left_cost + right_size * loss(tree.dist)
            cost_right_center = right_cost + left_size * loss(tree.dist)

            #We do not need to keep track of whether we are in the root.
            if cost_left_center > cost_right_center: #C_L > C_R implies higher decrease for C_R
                cost_decrease = loss(parent_dist) * total_size - cost_right_center 
                array.append((cost_decrease, right_center, tree))
                #print("Annotation:", cost_decrease, right_center, self.get_leaves(tree))

                return cost_right_center, right_center, total_size
            else:
                cost_decrease = loss(parent_dist) * total_size - cost_left_center 
                array.append((cost_decrease, left_center, tree))
                #print("Annotation:", cost_decrease, left_center,  self.get_leaves(tree))

                return cost_left_center, left_center, total_size
            
    if self.loss == "kmedian":
       annotation_builder(tree, output, np.inf, lambda x: x) #Using K-median loss
    elif self.loss == "kmeans":
       annotation_builder(tree, output, np.inf, lambda x: x**2) #Using K-means loss
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
  
    
  def prune_annotations(self, annotations):
     '''
     This removes any centers that only have a path of length 1 before another center trumps it in cost-decrease.
     It returns the pruned list of annotations.
     Since we use a counter instead of a boolean this is easily extendable to prune paths of length < l. 
     Annotations are built up as 
     '''

     annotations.sort(reverse=True, key=lambda x : x[1]) #Sort by the centers

     to_remove = set()
     #print("annotations:", [annotation[1] for annotation in annotations])
     curr_center = annotations[0][1]
     ctr = 0
     for annotation in annotations[1:]:
        new_center = annotation[1]
        if new_center != curr_center and ctr == 0:
           to_remove.add(curr_center)
           #print("removed", curr_center)
           curr_center = new_center
        elif new_center != curr_center:
           curr_center = new_center
           ctr = 0
        else:
           ctr += 1
     if ctr == 0:
        #print("removed", curr_center)
        to_remove.add(curr_center)
     new_annotations = [annotation for annotation in annotations if annotation[1] not in to_remove]
     #print("Anno size:", len(new_annotations))
     #print("annotations:", [annotation[1] for annotation in new_annotations])
     return new_annotations
  
  def prune_annotations_other(self, tree, annotations):
     '''
     The annotations are built up as a list of [(cost-decrease, center, tree),...]
     '''
     #points, _ = self.prune_cluster_subtree(tree, self.min_pts) #This is one option... however this will not prune below splits, which might not be ideal. 
     points, _ = self.prune_cluster_subtree_aggressive(tree, self.min_pts)
     point_set = set(points)
     #print("point_set:", point_set)
     pruned_annotations = []
     for annotation in annotations:
        center = annotation[1]
        if center in point_set:
            pruned_annotations.append(annotation)
    
     #Sorting just for the print:
     pruned_annotations.sort(reverse=True, key=lambda x : x[1]) #Sort by the centers
     #print("Other size:", len(pruned_annotations))
     #print("annotations other:", [annotation[1] for annotation in pruned_annotations])
     return pruned_annotations
  

     

  def normalize_cluster_ordering(self, cluster_labels):
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
  

  
  def cluster_stability_experimental(self, tree, cdists):
     var,bar,dsize = self.cluster_statistics(tree)
     nodes = self.get_leaves(tree)                 
     cluster_sum_inv = np.sum(1/cdists[nodes])  
     cluster_sum = np.sum(cdists[nodes])
     max_cdist = np.max(cdists)
     other_sum = np.sum(max_cdist - cdists[nodes])  
     mu_offset = np.mean(cdists) #Use the mean of the core-dists as an offset to fix the 0 issues.   
     stability1 = cluster_sum_inv/(var + bar)  
     stability2 = cluster_sum_inv / (var**2 + mu_offset)  
     stability3 = cluster_sum_inv / var if var > 1e-10 else stability1  
     stability4 = cluster_sum_inv / (var/bar) if var > 1e-10 and bar > 0 else stability3
     pdist = tree.parent.dist if tree.parent is not None else np.inf
     dist = tree.dist if not tree.is_leaf else cdists[tree.point_id]
     stability5 = (cluster_sum_inv * (1/dist - 1/pdist)) /  (var/bar) if var > 1e-10 else stability1

     show_nodes = min(5, len(nodes))
     if stability5 == np.inf:
        return 0
     return stability5  
  def cluster_statistics(self, dc_tree):
     '''
     Computes the variance of the dc-distance matrix of the set of nodes and all subsets of it bottom up.
     Returns the variance, mean and size of the distance matrix (lower triangular).
     '''
     if dc_tree.is_leaf:
        return 0,0,0 #return var, bar, num_pairwise_dists
     else:
        lvar, lbar, l_dists_size = self.cluster_statistics(dc_tree.left_tree)
        rvar, rbar, r_dists_size = self.cluster_statistics(dc_tree.right_tree)
        
        new_var, new_bar, new_dists_size = self.merge_subtree_variance(dc_tree.dist, l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, dc_tree.left_tree.size, dc_tree.right_tree.size)
        return new_var, new_bar, new_dists_size
       
  def combined_variance(self, n,m,xvar,yvar,xbar,ybar):
     '''
     https://stats.stackexchange.com/questions/557469/difference-between-pooled-variance-and-combined-variance  
     Returns the combined size, combined mean and combined variance
     '''
     nc = n+m
     xc = self.combined_mean(n,m,xbar,ybar)
     vc = (n*(xvar+(xbar-xc)**2)+m*(yvar+(ybar-xc)**2))/(nc)
     return vc, xc, nc  
  def combined_mean(self, n,m,xbar,ybar):
     return (xbar*n+ybar*m)/(n+m)
  
  def merge_subtree_variance(self, dist,  l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, lsetsize, rsetsize):
     if l_dists_size + r_dists_size != 0: 
        sub_var, sub_bar, sub_size = self.combined_variance(l_dists_size, r_dists_size, lvar, rvar, lbar, rbar)
        
        new_dists_size = lsetsize*rsetsize
        merge_var, merge_bar, merge_dists_size = self.combined_variance(sub_size, new_dists_size, sub_var, 0, sub_bar, dist)
     else: #If no dists in either distribution so far (coming from leaves)
        merge_var, merge_bar, merge_dists_size = 0, dist, 1  
     return merge_var, merge_bar, merge_dists_size
  
  def get_cdists(self, points, min_pts):
       '''
       Computes the core distances of a set of points, given a min_pts.
       '''
       num_points = points.shape[0]
       dim = int(points.shape[1])  
       D = np.zeros([num_points, num_points])
       D = self.get_dist_matrix(points, D, dim, num_points)  
       cdists = np.sort(D, axis=1)
       cdists = cdists[:, min_pts - 1] #These are the core-distances for each point.
       #print("cdists:", cdists)
       return cdists

  @staticmethod
  @numba.njit(fastmath=True, parallel=True)
  def get_dist_matrix(points, D, dim, num_points):
      '''
      Returns the Euclidean distance matrix of a 2D set of points.  
      Parameters
      ---------  
      points : n x m 2D numpy array
      D : empty n x n numpy array
      dim : m
      num_points : n
      '''
      for i in numba.prange(num_points):
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