from sklearn.datasets import make_swiss_roll, make_moons, make_blobs
from distance_metric import get_dc_dist_matrix, get_reach_dists
from sklearn.decomposition import KernelPCA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import numba as numba
from density_tree import make_tree
from tree_plotting import make_node_lists, find_node_positions
from datetime import datetime


def visualize_embedding(dists, names, distance, labels = None):
  '''
  Visualizes the distance measure in an embedded space using MDS
  '''
  fig, ax = plt.subplots()

  model = KernelPCA(n_components=2, kernel="precomputed")
  mds_embedding = model.fit_transform(-0.5 * dists)
  ax.scatter(mds_embedding[:, 0], mds_embedding[:, 1], c="b")
  for i, name in enumerate(names.values()):
     ax.annotate(name, (mds_embedding[i][0], mds_embedding[i][1]))
  
  ax.set_title("2D embedding of the distances with  " + distance + " distance")

  plt.show()

def visualize(points, cluster_labels = None, num_neighbors=None, embed = False, distance="dc_dist", minPts=3, show_cdists=False, save=False, save_name=None):
  '''
  Visualizes the complete graph G over the points with chosen distances on the edges.

  Parameters
  ----------
    points: The points to be visualized.
    cluster_labels: This shows a distinct color for each ground truth cluster a point is a part of.
    num_neighbors: Will be used to make graph less complete - only num_neighbors closest points will have edges. If none the graph will be complete
    embed: If true, will show the distances in embedded space with MDS.
    distance: The distance function to be used in the visualization. "dc_dist", "euclidean", "mut_reach".
    minPts: The number of points for a point to be a core point, determines core distance.
    show_cdists : Boolean, default=False
    save : Boolean, default=False
    save_name : String, default=None
  '''

  dists = get_dists(distance, points, minPts)
  
  fig, ax = plt.subplots(figsize=(16,9))
  ax.grid(True)
  ax.set_axisbelow(True)
  n = points.shape[0]
  G = nx.Graph()
  edges, edge_labels = create_edges(dists, num_neighbors)
  G.add_edges_from(edges)

  labels = {node: str(node) for node in G.nodes()}
  pos_dict = {i+1:(points[i,0], points[i,1]) for i in range(points.shape[0])}

  if cluster_labels is not None:
    nx.draw_networkx_nodes(G, pos=pos_dict, node_color=cluster_labels, ax=ax)
  else:
    nx.draw_networkx_nodes(G, pos=pos_dict, node_color="skyblue", ax=ax)

  nx.draw_networkx_labels(G, pos=pos_dict, labels=labels, font_color="black", ax=ax)
  nx.draw_networkx_edges(G, pos=pos_dict, ax=ax, width=0.8)
  nx.draw_networkx_edge_labels(G, pos=pos_dict, edge_labels=edge_labels, ax=ax, font_size=8)

  if show_cdists:
     cdists = get_cdists(points, minPts)


     for i, pos in pos_dict.items():
        circle = plt.Circle(pos, radius=cdists[i-1], edgecolor="black", facecolor="none")
        ax.add_patch(circle)

        edge_pos = (pos[0], pos[1]+cdists[i-1])
        ax.plot([pos[0], edge_pos[0]], [pos[1], edge_pos[1]], color='blue', zorder=0, alpha=0.5, linestyle='dotted')
        ax.text(pos[0], pos[1] + cdists[i-1]/2, str(np.round(cdists[i-1], 2)), ha='center', va='bottom', fontsize=6, color='black', rotation=90, bbox=None, zorder=1)

  #Set perspective to be "real"
  ax.set_aspect('equal', adjustable='box')
  #This is needed to add axis values to the plot
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

  ax.set_title("Complete graph with " + distance + " distance")
  
  #Visualize embedding
  if embed:
     visualize_embedding(dists, labels, distance)

  if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_graph.png")

  plt.show()



@numba.njit(fastmath=True, parallel=True)
def get_dist_matrix(points, D, dim, num_points):
    '''
    Returns the Euclidean distance matrix of a 2D set of points. 

    Parameters
    ----------

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

#Dist types: "Euclidean", "dc_dist", "mut_reach"
def get_dists(dist_type, points, minPts=3):
  '''
    Outputs the pairwise distance matrix between a set of points, each point being a row in a 2D array. 

    Parameters
    ----------
    dist_type : String
      Options: "euclidean", "dc_dist", "mut_reach"
    
    points : 2D numpy array

    minPts : Int, default=3
      The number of points for the core distance in dc_dist and mut_reach. The minimal number of points for something to be a core-point.
  '''
  dists = None
  n = points.shape[0]

  #Euclidean distance
  if dist_type == "euclidean":
    D = np.zeros([n, n])
    dists = get_dist_matrix(points, D, int(points.shape[1]), n)
  #dc-distance
  elif dist_type == "dc_dist":
    dists = get_dc_dist_matrix(
        points,
        n_neighbors=minPts, #Unused parameter
        min_points=minPts
    )
  #Mutual reachability distance
  elif dist_type == "mut_reach": 
    D = np.zeros([n, n])
    D = get_dist_matrix(points, D, int(points.shape[1]), n)
    dists = get_reach_dists(D, minPts, n)

  return dists
  
def print_numpy_code(array, newline=True):
   '''
    Prints a numpy array to the terminal in a format such that it can be directly copied into python code for quick experimentation with the same array.

    Parameters
    ----------
    array : A 2D numpy array
      
    newline : Boolean, default=True
      Will print each row on a separate line if true. Otherwise everything on a single line.
   '''
   if not newline:
    print("np.array([", end="")
   else:
    print("np.array([")
   for j, row in enumerate(array):
      print("[", end="")
      for i, elem in enumerate(row):
         if i == row.shape[0]-1:
            print(elem, end="")
         else:
            print(str(elem) + ",", end="")
      if j == array.shape[0]-1:
        print("]", end="")
      else:
        if not newline:
          print("],", end="")
        else:
          print("],")
   print("])")

#points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])





#TODO: Add so that I can see Euclidean, Mut Reach, DC and embedding at once
#TODO: Add so that colors are determined by cluster labelling
#TODO: Add plot titles that give information about what we are looking at
#TODO: Add cdist circles: TODOING
    
def get_cdists(points, min_pts):
    '''
    Computes the core distances of a set of points, given a min_pts.
    '''
    num_points = points.shape[0]
    dim = int(points.shape[1])

    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)

    cdists = np.sort(D, axis=1)
    cdists = cdists[:, min_pts - 1] #These are the core-distances for each point.
    print("cdists:", cdists)
    return cdists



#Currently creates edges for a complete graph
def create_edges(distance_matrix, num_neighbors = None):
  #print("dist matrix:", distance_matrix)
  edges = []
  edge_labels = {}
  for i in range(0, distance_matrix.shape[0]-1):
     for j in range(i+1,distance_matrix.shape[1]):
        edges.append((i+1,j+1))
        edge_labels[(i+1,j+1)] = np.round(distance_matrix[i,j],2)
  return edges, edge_labels

def main():
    samples = 6
    minPts = 2
    #Choose point distribution
    points, labels = make_moons(n_samples=samples, noise=0.1)
    #points, labels = make_blobs(n_samples=samples, centers=2)


    # root, dc_dists = make_tree(
    #     points,
    #     labels,
    #     min_points=minPts,
    #     make_image=True,
    #     n_neighbors=minPts
    # )




    #Choose distance function
    dist = "mut_reach"
    #dist = "euclidean"
    #dist = "dc_dist"


    print("points: \n", points)

    #get_cdists(points, 3)
    visualize(points=points, cluster_labels=labels, embed=True, distance=dist, minPts=minPts, show_cdists=True)
    #print_numpy_code(points)

main()