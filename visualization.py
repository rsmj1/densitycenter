from sklearn.datasets import make_swiss_roll, make_moons, make_blobs
from distance_metric import get_dc_dist_matrix, get_reach_dists
from sklearn.decomposition import KernelPCA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import numba as numba
from density_tree import make_tree



#Visualizes the distance measure in an embedded space using MDS
def visualize_embedding(dists, names, distance, labels = None):
  fig, ax = plt.subplots()

  model = KernelPCA(n_components=2, kernel="precomputed")
  mds_embedding = model.fit_transform(-0.5 * dists)
  ax.scatter(mds_embedding[:, 0], mds_embedding[:, 1], c="b")
  for i, name in enumerate(names.values()):
     ax.annotate(name, (mds_embedding[i][0], mds_embedding[i][1]))
  
  ax.set_title("2D embedding of the distances with  " + distance + " distance")

  plt.show()

'''
Visualizes the points with distances on the edges

Parameters:
  points: The points to be visualized.
  cluster_labels: This shows a distinct color for each ground truth cluster a point is a part of.
  num_neighbors: Will be used to make graph less complete - only num_neighbors closest points will have edges. If none the graph will be complete
  embed: If true, will show the distances in embedded space with MDS.
  distance: The distance function to be used in the visualization.
  minPts: The number of points for a point to be a core point, determines core distance.
'''
def visualize(points, cluster_labels = None, num_neighbors=None, embed = False, distance="dc_dist", minPts=3):
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

  #Set perspective to be "real"
  ax.set_aspect('equal', adjustable='box')
  #This is needed to add axis values to the plot
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

  ax.set_title("Complete graph with " + distance + " distance")
  
  #Visualize embedding
  if embed:
     visualize_embedding(dists, labels, distance)

  plt.show()



@numba.njit(fastmath=True, parallel=True)
def get_dist_matrix(points, D, dim, num_points):
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
  



#TODO: Add so that I can see Euclidean, Mut Reach, DC and embedding at once
#TODO: Add so that colors are determined by cluster labelling
#TODO: Add plot titles that give information about what we are looking at
#TODO: Add cdist circles

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
    minPts = 3
    #Choose point distribution
    points, labels = make_moons(n_samples=samples, noise=0.1)
    #points, labels = make_blobs(n_samples=samples, centers=2)


    root, dc_dists = make_tree(
        points,
        labels,
        min_points=minPts,
        make_image=True,
        n_neighbors=minPts
    )




    #Choose distance function
    dist = "mut_reach"
    #dist = "euclidean"
    #dist = "dc_dist"


    print("points: \n", points)


    visualize(points=points, cluster_labels=labels, embed=True, distance=dist, minPts=minPts)

main()