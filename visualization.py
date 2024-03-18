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
from matplotlib.widgets import Button, RadioButtons, CheckButtons


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

def visualize(points, cluster_labels = None, num_neighbors=None, embed = False, distance="dc_dist", minPts=3, centers=None, save=False, save_name=None):
  '''
  Visualizes the complete graph G over the points with chosen distances on the edges.

  Parameters
  ----------
    points: Numpy.Array
      The points to be visualized in a numpy 2D array

    cluster_labels: Numpy.Array, default=None 
      This shows a distinct color for each ground truth cluster a point is a part of, should be a 1D array with a label for each point. If None, will make all points blue.

    num_neighbors: Int, default=None
      Will be used to make graph less complete - only num_neighbors closest points will have edges. If none the graph will be complete
    
    embed : Boolean, default=False
      If true, will show the distances in embedded space with MDS.
    
    distance : String, default="dc_dist"
      The distance function to be used in the visualization. "dc_dist", "euclidean", "mut_reach".
    
    minPts: Int, default=3
      The number of points for a point to be a core point, determines core distance.

    show_cdists : Boolean, default=False
      Will make a circle with cdist radius from each point if true. Will have a dotted line to circle's edge showing the cdist value.
    
    centers : Numpy.Array, default=None
      Positions of the centers to be highlighted
    
    save : Boolean, default=False
      If true will save the plot under the save_name.
    
    save_name : String, default=None
      The name to save the plot under.
  '''

  #TODO: Add cdists again
  cdist_entities = []
  cdists_visible = False

  def toggle_cdists(event):
    nonlocal cdist_entities, cdists_visible
    if not cdists_visible:
       #Draw cdists
       for i, pos in pos_dict.items():
        circle = plt.Circle(pos, radius=cdists[i-1], edgecolor="black", facecolor="none", alpha=0.5)
        ax.add_patch(circle)
        cdist_entities.append(circle) #Save the circles to possibly destroy them

        edge_pos = (pos[0], pos[1]+cdists[i-1])
        line = ax.plot([pos[0], edge_pos[0]], [pos[1], edge_pos[1]], color='blue', zorder=0, alpha=0.5, linestyle='dotted')[0]
        text = ax.text(pos[0], pos[1] + cdists[i-1]/2, str(np.round(cdists[i-1], 2)), ha='center', va='bottom', fontsize=6, color='black', rotation=90, bbox=None, zorder=1)
        cdist_entities.append(line)
        cdist_entities.append(text)

        plt.draw()
        cdists_visible = True
    else:
       #Destroy cdists
       for c in cdist_entities:
          c.remove()
       cdist_entities = []
       plt.draw()
       cdists_visible = False

  cdists = get_cdists(points, minPts)
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



     
  #Code to highlight potential centers
  if centers is not None:
    print("highligthing centers")
    ax.scatter(centers[:, 0], centers[:,1], c="none", edgecolor="r", zorder=2, s=300)
     
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

  # Get the position and size of the plot's axes
  pos = ax.get_position()

  # Set the position and size of the button relative to the plot's axes
  button_width = 0.075
  button_height = 0.05
  button_spacing = 0.1
  button_x = pos.x0 - button_spacing
  button_y = pos.y0 + (pos.height - button_height) / 2 #This places it in the middle currently

  # Create a button
  button_ax = fig.add_axes([button_x, button_y, button_width, button_height]) #This defines the area on the plot that should be activated by the button widget
  cdist_button = Button(button_ax, 'Toggle cdists')

  # Attach the function to be called when the button is pressed
  cdist_button.on_clicked(toggle_cdists)


  plt.show()

#TODO: Add interactive button activating / deactivating cdists. use matplotlib.widgets.Button. Should be able to give it arguments. Need to keep the circles in a data structure, otherwise cannot disable cdists again. 
#TODO: Make it possible to change the distances on the edges between the different ones also via buttons, in this case Radiobuttons.
#TODO: Make on hover info about distances. https://stackoverflow.com/questions/61604636/adding-tooltip-for-nodes-in-python-networkx-graph
  
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
    print("points = np.array([", end="")
   else:
    print("points = np.array([")
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
    #print("cdists:", cdists)
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

#main()