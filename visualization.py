from sklearn.datasets import make_swiss_roll, make_moons
from distance_metric import get_dc_dist_matrix, get_reach_dists
from sklearn.decomposition import KernelPCA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import numba as numba



#Visualizes the distance measure in an embedded space using MDS
def visualize_embedding(dists, ax, names, labels = None):
  model = KernelPCA(n_components=2, kernel="precomputed")
  mds_embedding = model.fit_transform(-0.5 * dists)
  ax.scatter(mds_embedding[:, 0], mds_embedding[:, 1], c="b")
  for i, name in enumerate(names.values()):
     ax.annotate(name, (mds_embedding[i][0], mds_embedding[i][1]))

#Visualizes the points with distances on the edges
#The number of neigbohrs determines how many of the closest edges should be added. If None then the graph will be complete
'''
Parameters:
  points: The points to be visualized
  labels: If any default labels are with the points, these will be displayed in each node
'''
def visualize(points, labels = None, num_neighbors=None, embed = False, distance="dc_dist", minPts=3):
  dists = get_dists(distance, points, minPts)
  
  fig, (ax, ax2) = plt.subplots(1,2, figsize=(20,10))

  n = points.shape[0]
  G = nx.Graph()
  edges, edge_labels = create_edges(dists, num_neighbors)
  G.add_edges_from(edges)
  if labels is None:
    labels = {node: str(node) for node in G.nodes()}
  print("points:", points)
  pos_dict = {i+1:(points[i,0], points[i,1]) for i in range(points.shape[0])}

  nx.draw_networkx_nodes(G, pos=pos_dict, node_color="skyblue", ax=ax)
  nx.draw_networkx_labels(G, pos=pos_dict, labels=labels, font_color="black", ax=ax)

  nx.draw_networkx_edges(G, pos=pos_dict, ax=ax)
  nx.draw_networkx_edge_labels(G, pos=pos_dict, edge_labels=edge_labels, ax=ax)

  ax.set_aspect('equal', adjustable='box')
  #This is needed to add axis values to the plot
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

  ax.set_title("Complete graph with " + distance)

  #Visualize embedding
  if not embed:
    ax2.axis('off')
  else:
     visualize_embedding(dists, ax2, labels)
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
def get_dists(dist_type, points, minPts=None):
  dists = None
  n = points.shape[0]

  #Euclidean distance
  if dist_type == "Euclidean":
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
    points, labels = make_moons(n_samples=5, noise=0.1)

    visualize(points, None, embed=True)

main()