import efficientdcdist.dctree as dcdist
import numpy as np
import heapq
from density_tree import make_tree
from cluster_tree import prune_tree, copy_tree
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
    self.efficient_greedy(points)
    #self.simple_greedy(points)


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

  def label_points_tuples(self, points, dc_tree):
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
        Returns True from a single-center path and False otherwise
        '''
        if dc_tree.is_leaf:
            if dc_tree.center_path:
              print("here2")

              #list.append((dc_tree.point_id, dc_tree.unique_center))
              return True #First list is points we don't know what should be assigned to, second is those we know which center they belong to
            else:
               return False
        else:
            left_path= list_builder(dc_tree.left_tree, list)
            right_path = list_builder(dc_tree.right_tree, list)

            if dc_tree.center_path: #On center path they should be assigned
               print("here1")
               if dc_tree.num_centers == 1:
                  return True
               else: #Not uniquely close to one center
                  #Now, those points that cmoe from a single-center path should now be pruned together via external method.
                  #Those that don't should be assigned to noise
                  if left_path:
                     points, noise = self.prune_cluster_subtree(dc_tree.left_tree, self.min_pts)
                  else:
                     points = self.get_leaves(dc_tree.left_tree)
                     #Assign to noise
                  if right_path:
                     points, noise = self.prune_cluster_subtree(dc_tree.right_tree, self.min_pts)

                  else:
                     points = self.get_leaves(dc_tree.right_tree)

                  return
               return
            else: #Not on center path - cannnot be assigned yet
               return False


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


  def prune_cluster_subtree(self, dc_tree, min_pts):
   '''
   Will prune the tree and return the points that should be pruned and those that should be assigned.
   '''
   pruned_tree = prune_tree(dc_tree)

   def noise_collector(tree, points_list, noise_list):
      
      return

   

   return noise_collector(pruned_tree, [],[])




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


  def assign_points_prune_v2(self, points, dc_tree, min_pts):
     '''
     This method will both assign points that are equidistant to two clusters and branch-off points to noise. 
     #TODO: Need to understand how this should work in the first place...
     '''
     return

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

     #This is to avoid picking noise points as centers
     annotations = self.prune_annotations(annotations)

     annotations.sort(reverse=True, key=lambda x : x[0]) #Sort by the first value of the tuples - the potential cost-decrease. Reverse=True to get descending order.
     cluster_centers = set() #We should not use the "in" operation, as it is a worst-case O(n) operation. Just add again and again

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
     #self.labels_ = self.assign_points_eff(points, dc_tree)
     
     centers = list(cluster_centers)


     self.mark_paths(dc_tree, centers)
     self.labels_ = self.assign_points_prune(points, dc_tree)
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
     '''

     output = []
     def annotation_builder(tree, array, parent_dist):
        if tree.is_leaf:
          array.append((parent_dist, tree.point_id, tree)) #Need to append a pointer to the tree itself because we want to mark the spots in the tree form which a point was chosen.
          #print("Annotation:", parent_dist, tree.point_id,  self.get_leaves(tree))
          return 0, tree.point_id, 1 #Returns local cost, the center for that cost and the size of the tree
        else:
           left_cost, left_center, left_size = annotation_builder(tree.left_tree, array, tree.dist)
           right_cost, right_center, right_size =  annotation_builder(tree.right_tree, array, tree.dist)
           total_size = left_size + right_size

           cost_left_center = left_cost + right_size * tree.dist
           cost_right_center = right_cost + left_size * tree.dist

          #We do not need to keep track of whether we are in the root.
           if cost_left_center > cost_right_center: #C_L > C_R implies higher decrease for C_R
              cost_decrease = parent_dist* total_size - cost_right_center 
              array.append((cost_decrease, right_center, tree))
              #print("Annotation:", cost_decrease, right_center, self.get_leaves(tree))

              return cost_right_center, right_center, total_size
           else:
              cost_decrease = parent_dist* total_size - cost_left_center 
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
  




  
  def prune_annotations(self, annotations):
     '''
     This removes any centers that only have a path of length 1 before another center trumps it in cost-decrease.
     It returns the pruned list of annotations.
     Since we use a counter instead of a boolean this is easily extendable to prune paths of length < l. 
     '''

     annotations.sort(reverse=True, key=lambda x : x[1]) #Sort by the centers

     to_remove = set()
     print("annotations:", [annotation[1] for annotation in annotations])
     curr_center = annotations[0][1]
     ctr = 0
     for annotation in annotations[1:]:
        new_center = annotation[1]
        if new_center != curr_center and ctr == 0:
           to_remove.add(curr_center)
           print("removed", curr_center)
           curr_center = new_center
        elif new_center != curr_center:
           curr_center = new_center
           ctr = 0
        else:
           ctr += 1
     if ctr == 0:
        print("removed", curr_center)
        to_remove.add(curr_center)
     new_annotations = [annotation for annotation in annotations if annotation[1] not in to_remove]
     print("annotations:", [annotation[1] for annotation in new_annotations])

     return new_annotations





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
