import efficientdcdist.dctree as dcdist
import numpy as np
import heapq
import numba
from n_density_tree import make_n_tree, prune_n_tree

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
        dc_tree, _ = make_n_tree(points, placeholder, min_points=self.min_pts, )

        annotations = self.annotate_tree(dc_tree) #Will use K-means or K-median loss depending on self.loss


        if self.noise_mode == "medium" or self.noise_mode == "full":
                #This is to avoid picking noise points as centers
                #annotations1 = self.prune_annotations_other(dc_tree, annotations) #This prunes based on the pruned tree. This is always slightly less aggressive than the one below. 
                annotations = self.prune_annotations(annotations) #This prunes based on occurences in the annotation list.

        annotations.sort(reverse=True, key=lambda x : x[0]) #Sort by the first value of the tuples - the potential cost-decrease. Reverse=True to get descending order.
        print("annotations NARY:", [(x[1]) for x in annotations])

        
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
        centers = list(cluster_centers)
        if self.noise_mode == "none":
            self.labels_ = self.assign_points_eff(points, dc_tree)
        elif self.noise_mode == "medium":
            self.mark_center_paths(dc_tree, centers)
            self.labels_ = self.assign_points_prune(points, dc_tree)
        elif self.noise_mode == "full":
            self.mark_center_paths(dc_tree, centers)
            self.labels_ = self.assign_points_prune_full(points, dc_tree) 
            #self.labels_ = self.assign_points_prune_stability(dc_tree, self.cluster_stability_experimental)      
            
        else: 
            raise AssertionError("The noise detection mode is not recognized. Choose between none, medium or full.")
        
        print("labels N-ary:", self.labels_)

        self.center_indexes = centers
        self.centers = points[centers]
        return

    def simple_greedy(self, points):
        '''
        Solves the K-median / K-means problem optimally over the dc-distance with O(n^2k^2) complexity.
        It does so by greedily choosing the next center as the point that reduces the cost the most. 
        '''
        n = points.shape[0]
        dc_tree = dcdist.DCTree(points, min_points=self.min_pts, n_jobs=1) #This is the efficient implementation of the dc_tree for measuring distances. 
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
                for child in dc_tree.children:
                    list_builder(child, list, center)

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
                if dc_tree.center_path: #We don't need to check how many centers here, can only be 1.
                    list.append((dc_tree.point_id, dc_tree.unique_center))
                    return []
                else:
                    return [dc_tree.point_id]
            else:
                all_not_assigned = []
                for child in dc_tree.children:
                    all_not_assigned += list_builder(child, list)

                if dc_tree.center_path: #On center path they should be assigned
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
                if dc_tree.center_path:
                    if dc_tree.num_centers == 1:
                        return 0
                    else: #Not uniquely close to one center
                        #Now, those points that come from a single-center path should now be pruned together via external method.
                        #Those that don't should be assigned to noise
                        for child in dc_tree.children:
                            path = list_builder(child, list)
                            if path == 0: #Assign via pruned tree
                                points, noise = self.prune_cluster_subtree(child, self.min_pts)
                                center = child.unique_center
                                for point in points:
                                    list.append((point, center))  
                                for point in noise:
                                    list.append((point, -1))
                            elif path == 1: #Assign to noise
                                noise = self.get_leaves(child)
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

                    return 0, stability(dc_tree, self.cdists), dc_tree
                else:
                    return 1, 0, None #Not a center path - so no need to compute a stability and return any pointer to a tree.
            else:
                if dc_tree.center_path: #On center path they should be assigned
                    if dc_tree.num_centers == 1: #Compute new stability since we are on a center path and return the best of the stabilities.
                        new_stability = stability(dc_tree, self.cdists)
                        for child in dc_tree.children:
                            path, below_stability, below_cluster = list_builder(child, list, stability)
                            if path == 0: #We can only have one path below returning 0, so we just find this path.
                                if new_stability >= below_stability:
                                    return 0, new_stability, dc_tree
                                else:
                                    return 0, below_stability, below_cluster
                        
                    else: #Not uniquely close to one center (potentially anymore, we might be coming from a single center path)
                        for child in dc_tree.children:
                            path, below_stability, below_cluster = list_builder(child, list, stability)

                            if path == 0:
                                points = self.get_leaves(below_cluster)
                                center = child.unique_center
                                noise = [p for p in self.get_leaves(child) if p not in points]
                                for point in points:
                                    list.append((point, center))  
                                for point in noise:
                                    list.append((point, -1))
                            elif path == 1:
                                noise = self.get_leaves(child)
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
        It will return points below splits in the pruned tree and the actual leaves of the pruned tree as non-noise points. 
        '''
        pruned_tree = prune_n_tree(dc_tree, min_pts)

        def prune_collector(tree, points_list):
            if tree.is_leaf:
                points_list.append(tree.point_id)
            else:
                if self.is_split(tree):
                    for child in tree.children: #Only add the split points to the set of points, not the potential pruned things at that level.
                        points_list += list(self.get_leaves(child.orig_node))
                    return
                
                for child in tree.children:
                    prune_collector(child, points_list)

        points, noise = [], []
        if pruned_tree is None: #If everything is pruned - then it should return the nodes - it does not make sense that everything is noise.. TODO
            points += self.get_leaves(dc_tree)
        else:
            prune_collector(pruned_tree, points)
            noise = [p for p in self.get_leaves(dc_tree) if p not in points]
        return points, noise

    def is_split(self, dc_tree):
        '''
        Detects splits in a tree. Mainly used on a pruned tree where you can have less than two children per node. 
        '''
        num_splits = 0
        for child in dc_tree.children: 
            if child.point_id is None: #If these ids are none they are internal nodes - ergo we have a split in the pruned tree.
                num_splits += 1
        return True if num_splits >= 2 else False


    def prune_cluster_subtree_aggressive(self, dc_tree, min_pts):
        '''
        Will prune the tree and return the points that should be pruned and those that should be assigned.
        This keeps going to the leaves of the pruned tree no matter whether there is a split in the pruned tree or not. 
        '''
        pruned_tree = prune_n_tree(dc_tree, min_pts)
        if pruned_tree is None:
            return list(self.get_leaves(dc_tree)), []
        all_points = self.get_leaves(dc_tree)
        points = list(self.get_leaves(pruned_tree))
        noise  = [p for p in all_points if p not in points]
    
        return points, noise

    def mark_center_paths(self, dc_tree, centers):
        '''
        Inefficient implementation - it is n^2
        This marks in the tree the paths from the centers and how many centers are on overlapping paths.
        This is used for detecting when points are equidistant to multiple centers. 
        '''
        if dc_tree.is_leaf:
            if dc_tree.point_id in centers:
                dc_tree.center_path = True
                dc_tree.num_centers = 1
                dc_tree.unique_center = dc_tree.point_id
                return [dc_tree.point_id]
            return []
        else:
            union_path = []
            for child in dc_tree.children:
                union_path += self.mark_center_paths(child, centers)

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


    def annotate_tree(self, tree): #Modified
        '''
        Returns the list of annotations. 
        '''
        output = []
        def annotation_builder(tree, array, parent_dist, loss):
            if tree.is_leaf:
                array.append((loss(parent_dist), tree.point_id, tree)) #Need to append a pointer to the tree itself because we want to mark the spots in the tree form which a point was chosen.
                #print("Annotation:", parent_dist, tree.point_id,  self.get_leaves(tree))
                return 0, tree.point_id #Returns local cost, the center for that cost and the size of the tree
            else:
                
                best_cost, best_center = np.inf,-1
                for child in tree.children:
                    curr_cost, curr_center = annotation_builder(child, array, tree.dist, loss)
                    cost_curr_center = curr_cost + loss(tree.dist) * (tree.size - child.size)
                    if cost_curr_center <= best_cost:
                        best_cost, best_center= cost_curr_center, curr_center
                
                best_cost_decrease = loss(parent_dist) * tree.size - best_cost
                array.append((best_cost_decrease, best_center, tree))
                
                return best_cost, best_center
            
                
        if self.loss == "kmedian":
            annotation_builder(tree, output, np.inf, lambda x: x) #Using K-median loss
        elif self.loss == "kmeans":
            annotation_builder(tree, output, np.inf, lambda x: x**2) #Using K-means loss
        return output



    def get_leaves(self, dc_tree):
            '''
            Returns the set of ids of the leaf nodes within the given cluster.
            '''
            def leaf_helper(dc_tree):
                if dc_tree.is_leaf:
                    return [dc_tree.point_id]
                else:
                    leaves = []
                    for child in dc_tree.children:
                        leaves += leaf_helper(child)
                    return leaves
                
            return np.array(leaf_helper(dc_tree))
    
        
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
        points = self.prune_cluster_subtree_aggressive(tree, self.min_pts)
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
                total_var, total_bar, total_dists_size = self.cluster_statistics(dc_tree.children[0])
                total_tree_size = dc_tree.children[0].size
                for child in dc_tree.children[1:]:
                    new_var, new_bar, new_dists_size = self.cluster_statistics(child)
                    total_var, total_bar, total_dists_size = self.merge_subtree_variance(dc_tree.dist, total_dists_size, new_dists_size, total_var, new_var, total_bar, new_bar, total_tree_size, child.size)
                    total_tree_size += child.size
                
                return total_var, total_bar, total_dists_size
        
 
    def merge_subtree_variance(self, dist,  l_dists_size, r_dists_size, lvar, rvar, lbar, rbar, lsetsize, rsetsize):
        if l_dists_size + r_dists_size != 0: 
            sub_var, sub_bar, sub_size = self.combined_variance(l_dists_size, r_dists_size, lvar, rvar, lbar, rbar)
            
            new_dists_size = lsetsize*rsetsize
            merge_var, merge_bar, merge_dists_size = self.combined_variance(sub_size, new_dists_size, sub_var, 0, sub_bar, dist)
        else: #If no dists in either distribution so far (coming from leaves)
            merge_var, merge_bar, merge_dists_size = 0, dist, 1

        return merge_var, merge_bar, merge_dists_size

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