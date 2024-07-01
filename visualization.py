from sklearn.datasets import make_swiss_roll, make_moons, make_blobs
from distance_metric import get_dc_dist_matrix, get_reach_dists
from sklearn.decomposition import KernelPCA
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numba as numba
from datetime import datetime
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from density_tree import DensityTree
from density_tree_nary import NaryDensityTree




def plot_embedding(embed_points, embed_labels, titles, centers = None, main_title=None, dot_scale = 1, annotations=False, save=False, save_name=None):
    '''
    Plots the provided points with as many embeddings as there are sets of lables and titles. Will highlight centers in k-means embeddings if provided.

    Parameters
    ----------

    embed_points : Numpy.Array
    embed_labels : List[Numpy.Array]
    titles : List[String]
    centers : Numpy.Array, default=None
    dot_scale : Float, default=1
    save : Boolean, default=False
    save_name : String, default=None
    '''
    dot_size = (plt.rcParams['lines.markersize'] ** 2) * dot_scale

    if len(embed_points.shape) == 1:
        embed_points = np.stack((embed_points, np.zeros_like(embed_points)), -1)


    
    if not isinstance(embed_labels, list):
        embed_labels = [embed_labels]
    if not isinstance(titles, list):
        titles = [titles]
    assert len(embed_labels) == len(titles)
    if len(embed_labels) == 1:
        fig, axes = plt.subplots(1)
        axes = [axes]  # Convert single Axes object to a list
    else:
        fig, axes = plt.subplots(1, len(embed_labels))
        fig.set_figwidth(4 * len(embed_labels))

    co = 0 #This is a janky solution to only show centers on the one provided.

    for i, labels in enumerate(embed_labels):
        if not isinstance(labels[0], str):
            edgecolors = np.where(labels == -1, "yellow", "black")
        else:
            edgecolors = ["black" for x in embed_labels]

        noise_points = np.array([point for i,point in enumerate(embed_points) if labels[i] == -1])
        noise_labels = ["lightgrey" for point in noise_points]
        noise_edgecolors = ["lightgrey" for point in noise_points]

        #"Paired" is good and used for compound
        cmap = "Paired" #Default is "viridis", other options are https://matplotlib.org/stable/users/explain/colors/colormaps.html
        axes[i].scatter(embed_points[:, 0], embed_points[:, 1], c=labels, cmap=cmap, s=dot_size, edgecolor=edgecolors, zorder=1)

        #Plot noise points on top
        if len(noise_points) != 0:
            axes[i].scatter(noise_points[:, 0], noise_points[:, 1], c=noise_labels, s=dot_size, edgecolor=noise_edgecolors, zorder=2)


        axes[i].set_title(titles[i], fontsize=20)
        if "K-means" in titles[i] and centers is not None and co == 0:
            axes[i].scatter(centers[:, 0], centers[:,1], c="none", s=dot_size, edgecolor="r")
            co += 1
        axes[i].grid(alpha=0.2, zorder=0)

        # Adding annotations for each point
        if annotations:
            for j, point in enumerate(embed_points):
                x, y = point
                axes[i].annotate(str(j+1), xy=(x, y), xytext=(0, 0), textcoords='offset points', ha='center', va='center')
        
        # Set the tick formatter to display integers
        # axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
        # axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    if main_title is not None:
        fig.suptitle(main_title)

    if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_embeddings.png")
    
    plt.show()




def visualize_embedding(points, labels=None, min_pts=3, metric="dc_dist"):
  '''
  Visualizes the distance measure in an embedded space using MDS
  '''
  fig, ax = plt.subplots()
  if labels is None:
    labels = np.arange(len(points))

  dists = get_dists(metric, points, min_pts=min_pts)

  model = KernelPCA(n_components=2, kernel="precomputed")
  mds_embedding = model.fit_transform(-0.5 * dists)
  ax.scatter(mds_embedding[:, 0], mds_embedding[:, 1], c="b")
  for i, name in enumerate(labels):
     ax.annotate(name, (mds_embedding[i][0], mds_embedding[i][1]))
  
  ax.set_title("2D embedding of the distances with  " + metric + " distance")

  plt.show()

def visualize(points, cluster_labels = None, distance="dc_dist", minPts=3, centers=None, save=False, save_name=None):
  '''
  Visualizes the complete graph G over the points with chosen distances on the edges.

  Parameters
  ----------
    points: Numpy.Array
      The points to be visualized in a numpy 2D array

    cluster_labels: Numpy.Array, default=None 
      This shows a distinct color for each ground truth cluster a point is a part of, should be a 1D array with a label for each point. If None, will make all points blue.
    
    distance : String, default="dc_dist"
      The distance function to be used in the visualization. "dc_dist", "euclidean", "mut_reach".
    
    minPts: Int, default=3
      The number of points for a point to be a core point, determines core distance.

    show_cdists : Boolean, default=False
      Will make a circle with cdist radius from each point if true. Will have a dotted line to circle's edge showing the cdist value.
    
    centers : Numpy.Array, default=None
      Positions of the centers to be highlighted. Needs 2d coordinates. 
    
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
  edges, edge_labels = create_edges(dists)
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
    #print("dc_dist_matrix:", np.round(dists,2))
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
def create_edges(distance_matrix):
  #print("dist matrix:", distance_matrix)
  edges = []
  edge_labels = {}
  for i in range(0, distance_matrix.shape[0]-1):
     for j in range(i+1,distance_matrix.shape[1]):
        edges.append((i+1,j+1))
        edge_labels[(i+1,j+1)] = np.round(distance_matrix[i,j],2)
  return edges, edge_labels


    ###########################################
    ######## N-ary tree plotting tools ########
    ###########################################


def find_node_positions_nary(root, width=1, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None):
    if pos is None:
        pos = [[xcenter, vert_loc]]
    else:
        pos.append([xcenter, vert_loc])
    if root.has_children:
        dx = width / root.num_children
        curr_x = xcenter - (dx*(root.num_children-1)/2)
        for child in root.children:
          pos = find_node_positions_nary(
             child,
             width = dx,
             vert_gap=vert_gap,
             vert_loc=vert_loc-vert_gap,
             xcenter=curr_x,
             pos=pos,
          )
          curr_x += dx
    return pos

def make_node_lists_nary(root, point_labels, parent_count, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers=None):
    count = parent_count
    if root.dist > 0:
        dist_list.append(root.dist)
    else: 
        dist_list.append(root.point_id+1)
    if root.is_leaf and root.point_id is not None:
        if root.point_id == -2:
            color_list.append(0)
        else:
            color_list.append(point_labels[root.point_id])
            alpha_list.append(1)
            if centers is not None:
                if root.point_id in centers:
                    edgecolor_list.append("red")
                elif point_labels[root.point_id] != -1:
                    edgecolor_list.append("black")
                else: 
                    edgecolor_list.append("yellow")
            else: 
                if point_labels[root.point_id] != -1: #Non-noise points
                    edgecolor_list.append("black")
                else: #Noise points:
                    edgecolor_list.append("yellow")
    else:
        color_list.append(-1)
        alpha_list.append(0)
        edgecolor_list.append("black")

    for tree in root.children:
        if tree is not None:
            edge_list.append((parent_count, count+1))
            count = make_node_lists_nary(
                tree,
                point_labels,
                count+1,
                dist_list,
                edge_list,
                color_list,
                alpha_list,
                edgecolor_list,
                dist_dict,
                centers
            )

    return count


def find_node_positions(root, width=1, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None):
    if pos is None:
        pos = [[xcenter, vert_loc]]
    else:
        pos.append([xcenter, vert_loc])
    if root.left_tree is not None and root.right_tree is not None:
        dx = width / 2
        left_x = xcenter - dx / 2
        right_x = left_x + dx
        pos = find_node_positions(
            root.left_tree,
            width=dx,
            vert_gap=vert_gap, 
            vert_loc=vert_loc-vert_gap,
            xcenter=left_x,
            pos=pos,
        )
        pos = find_node_positions(
            root.right_tree,
            width=dx,
            vert_gap=vert_gap, 
            vert_loc=vert_loc-vert_gap,
            xcenter=right_x,
            pos=pos,
        )

    return pos

def make_node_lists(root, point_labels, parent_count, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers=None):
    count = parent_count
    if root.dist > 0:
        dist_list.append(root.dist)
    else: 
        dist_list.append(root.point_id+1)
    if root.is_leaf:
        if root.point_id == -2:
            color_list.append(0)
        else:
            color_list.append(point_labels[root.point_id])
            alpha_list.append(1)
            if centers is not None:
                if root.point_id in centers:
                    edgecolor_list.append("red")
                elif point_labels[root.point_id] != -1:
                    edgecolor_list.append("black")
                else: 
                    edgecolor_list.append("yellow")
            else: 
                if point_labels[root.point_id] != -1: #Non-noise points
                    edgecolor_list.append("black")
                else: #Noise points:
                    edgecolor_list.append("yellow")
    else:
        color_list.append(-1)
        alpha_list.append(0.5)
        edgecolor_list.append("black")

    for tree in [root.left_tree, root.right_tree]:
        if tree is not None:
            edge_list.append((parent_count, count+1))
            count = make_node_lists(
                tree,
                point_labels,
                count+1,
                dist_list,
                edge_list,
                color_list,
                alpha_list,
                edgecolor_list,
                dist_dict,
                centers
            )

    return count


def plot_tree(root, labels=None, centers=None, save=False, save_name=None, is_binary=True, extra_annotations=None):
    '''
    ***THIS METHOD IS KEPT AS LEGACY AS IT STILL IS ABLE TO PLOT THE BINARY DC-TREE***

    Plots the dc-dist tree, optionally highligthing nodes chosen as centers with a red outline. Shows the node indexes on the leaves and dc-distances in the non-leaf nodes. The leaves are color-coded by the provided labels.
    A yellow outline means that a node was labelled noise. 
    Parameters
    ----------
    root : DensityTree
    labels : Numpy.Array
    centers : Numpy.Array, default=None
    save : Boolean, default=False
    save_name : String, default=None
    extra_annotations, Numpy.Array, default=None
      The annotations should be provided in preorder traversal order over the binary tree.
    '''

    if labels is None:
       labels = np.arange(root.size)
    dist_dict = {}

    edge_list = []
    dist_list = []
    color_list = []
    alpha_list = []
    edgecolor_list = []

    if is_binary:
      assert isinstance(root, DensityTree), "is_binary is True so expected a root of class DensityTree"
      make_node_lists(root, labels, 1, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers)
      pos_list = find_node_positions(root, 10)
    else:
      assert isinstance(root, NaryDensityTree), "is_binary is False so expected a root of class NaryDensityTree"
      make_node_lists_nary(root, labels, 1, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers)
      pos_list = find_node_positions_nary(root, 10)

    G = nx.Graph()
    G.add_edges_from(edge_list)

    if extra_annotations is not None and len(extra_annotations) != G.number_of_nodes():
      print("You should only use extra annotations with mcs <= 2!")
      raise AssertionError("Extra annotations are not compatible!")       
    print("G.nodes:", G.nodes)

    extra_dict = {}
    pos_dict = {}
    for i, node in enumerate(G.nodes):
        pos_dict[node] = pos_list[i]
        #+1 for {:.0f} as these are the node numbers which are 0 indexed from the point_ids in the tree, but are 1-indexed in the other visualizations.
        dist_dict[node] = '{:.2f}'.format(dist_list[i]) if dist_list[i] % 1 != 0 else '{:.0f}'.format(dist_list[i])

        if extra_annotations is not None: #Also for extra here
          extra_dict[node] = np.round(extra_annotations[i],2) 

    if extra_annotations is not None:
      #New modification for optional annotations on the tree here.
      for node, (x, y) in pos_dict.items():
          #print("Node:", node)
          #First two are the positions of the extra text, the third is the actual text to add.
          plt.text(x, y + 0.05, extra_dict[node], horizontalalignment='center', fontsize=8)

    if is_binary:  
      plt.title("binary dc-distance tree with " + str(len(labels)) + " points")
    else:
      plt.title("n-ary dc-distance tree with " + str(len(labels)) + " points")
    
    nx.draw_networkx_nodes(G, pos=pos_dict, node_color=color_list, alpha=alpha_list, edgecolors=edgecolor_list, linewidths=1.5, node_size=450)
    nx.draw_networkx_edges(G, pos=pos_dict)
    nx.draw_networkx_labels(G, pos=pos_dict, labels=dist_dict, font_size=8)
    
    if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_tree.png")

    plt.show()


def make_node_lists_nary_v2(root, point_labels, parent_count, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers=None):
    count = parent_count
    if root.dist > 0:
        dist_list.append(root.dist)
    else: 
        dist_list.append(root.point_id+1)
    if root.is_leaf and root.point_id is not None:
        if root.point_id == -2:
            color_list.append(0)
        else:
            color_list.append(point_labels[root.point_id])
            alpha_list.append(1)
            if centers is not None:
                if root.point_id in centers:
                    edgecolor_list.append("red")
                elif point_labels[root.point_id] != -1:
                    edgecolor_list.append("black")
                else: 
                    edgecolor_list.append("yellow")
            else: 
                if point_labels[root.point_id] != -1: #Non-noise points
                    edgecolor_list.append("black")
                else: #Noise points:
                    edgecolor_list.append("yellow")
    else: #Internal node
        color_list.append("white")
        alpha_list.append(1)
        edgecolor_list.append("black")

    for tree in root.children:
        if tree is not None:
            edge_list.append((parent_count, count+1))
            count = make_node_lists_nary_v2(
                tree,
                point_labels,
                count+1,
                dist_list,
                edge_list,
                color_list,
                alpha_list,
                edgecolor_list,
                dist_dict,
                centers
            )

    return count

def plot_tree_v2(root, labels=None, centers=None, save=False, save_name=None, extra_annotations=None, node_size=900):
    '''
    Plots the dc-dist tree, optionally highligthing nodes chosen as centers with a red outline. Shows the node indexes on the leaves and dc-distances in the non-leaf nodes. The leaves are color-coded by the provided labels.
    A yellow outline means that a node was labelled noise. 
    Parameters
    ----------
    root : NaryDensityTree
    labels : Numpy.Array
    centers : Numpy.Array, default=None
    save : Boolean, default=False
    save_name : String, default=None
    extra_annotations : Numpy.Array, default=None
      The annotations should be provided in preorder traversal order over the binary tree.
    node_size : int, default=900
    '''

    if labels is None:
       labels = np.arange(root.size)
    dist_dict = {}

    edge_list = []
    dist_list = []
    color_list = []
    alpha_list = []
    edgecolor_list = []

    assert isinstance(root, NaryDensityTree), "is_binary is False so expected a root of class NaryDensityTree"
    make_node_lists_nary_v2(root, labels, 1, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers)
    pos_list = find_node_positions_nary(root, 10)

    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    extra_dict = {}
    pos_dict = {}
    for i, node in enumerate(G.nodes):
        pos_dict[node] = pos_list[i]
        #+1 for {:.0f} as these are the node numbers which are 0 indexed from the point_ids in the tree, but are 1-indexed in the other visualizations.
        dist_dict[node] = '{:.1f}'.format(dist_list[i]) if dist_list[i] % 1 != 0 else '{:.0f}'.format(dist_list[i])

        if extra_annotations is not None: #Also for extra here
          extra_dict[node] = np.round(extra_annotations[i],2) 

    plt.title("n-ary dc-distance tree with " + str(len(labels)) + " points")



    # Identify internal nodes and leaf nodes
    internal_nodes = [node for node in G.nodes() if G.degree(node) > 1]
    leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]

    # Split color, alpha, and edgecolor lists
    internal_color_list = [color_list[node - 1] for node in internal_nodes]
    leaf_color_list = [color_list[node - 1] for node in leaf_nodes]
    #print("internal colors:", internal_color_list)
    #print("leaf_color_list:", leaf_color_list)

    internal_alpha_list = [alpha_list[node - 1] for node in internal_nodes]
    leaf_alpha_list = [alpha_list[node - 1] for node in leaf_nodes]

    internal_edgecolor_list = [edgecolor_list[node - 1] for node in internal_nodes]
    leaf_edgecolor_list = [edgecolor_list[node - 1] for node in leaf_nodes]


    noise_nodes = [node for node in G.nodes() if (G.degree(node) == 1 and color_list[node - 1] == -1)]
    noise_color_list = ["lightgrey" for node in noise_nodes]
    noise_alpha_list = [alpha_list[node - 1] for node in noise_nodes]
    noise_edgecolor_list = ["lightgrey" for node in noise_nodes]


    
    # Draw internal nodes
    nx.draw_networkx_nodes(G, pos=pos_dict, nodelist=internal_nodes, node_color=internal_color_list, alpha=internal_alpha_list, edgecolors=internal_edgecolor_list, linewidths=1.5, node_size=node_size)

    cmap = "Paired"
    # Draw leaf nodes
    nx.draw_networkx_nodes(G, pos=pos_dict, nodelist=leaf_nodes, node_color=leaf_color_list, alpha=leaf_alpha_list, edgecolors=leaf_edgecolor_list, linewidths=1.5, node_size=node_size, cmap=cmap)
    
    # Draw noise nodes
    if len(noise_nodes) != 0:
      nx.draw_networkx_nodes(G, pos=pos_dict, nodelist=noise_nodes, node_color=noise_color_list, alpha=noise_alpha_list, edgecolors=noise_edgecolor_list, linewidths=1.5, node_size=node_size)


    nx.draw_networkx_edges(G, pos=pos_dict)
    nx.draw_networkx_labels(G, pos=pos_dict, labels=dist_dict, font_size=max(6,int(node_size / 75)))
    

    if extra_annotations is not None:
      #New modification for optional annotations on the tree here.
      for node, (x, y) in pos_dict.items():
          #print("Node:", node)
          #First two are the positions of the extra text, the third is the actual text to add.
          val = 0
          if extra_dict[node] != 0.0:
             val = extra_dict[node]

          plt.text(x, y + 0.05, val, horizontalalignment='center', fontsize=max(6,int(node_size / 75)), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))



    if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_tree.png")

    plt.show()
