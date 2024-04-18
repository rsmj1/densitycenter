import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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

def plot_tree(root, labels, centers=None, save=False, save_name=None):
    '''
    Plots the dc-dist tree, optionally highligthing nodes chosen as centers with a red outline. Shows the node indexes on the leaves and dc-distances in the non-leaf nodes. The leaves are color-coded by the provided labels.
    A yellow outline means that a node was labelled noise. 
    Parameters
    ----------

    root : DensityTree
    labels : Numpy.Array
    centers : Numpy.Array, default=None
    save : Boolean, default=False
    save_name : String, default=None
    '''
    dist_dict = {}

    edge_list = []
    dist_list = []
    color_list = []
    alpha_list = []
    edgecolor_list = []
    make_node_lists(root, labels, 1, dist_list, edge_list, color_list, alpha_list, edgecolor_list, dist_dict, centers)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos_list = find_node_positions(root, 10)

    pos_dict = {}
    for i, node in enumerate(G.nodes):
        pos_dict[node] = pos_list[i]
        #+1 for {:.0f} as these are the node numbers which are 0 indexed from the point_ids in the tree, but are 1-indexed in the other visualizations.
        dist_dict[node] = '{:.2f}'.format(dist_list[i]) if dist_list[i] % 1 != 0 else '{:.0f}'.format(dist_list[i])

    
    plt.title("dc-distance tree with n=" + str(len(labels)))
    
    nx.draw_networkx_nodes(G, pos=pos_dict, node_color=color_list, alpha=alpha_list, edgecolors=edgecolor_list, linewidths=1.5)
    nx.draw_networkx_edges(G, pos=pos_dict)
    nx.draw_networkx_labels(G, pos=pos_dict, labels=dist_dict, font_size=8)
    

    if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_tree.png")

    plt.show()


def plot_embedding(embed_points, embed_labels, titles, centers = None, main_title=None, dot_scale = 1, save=False, save_name=None):
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
    fig, axes = plt.subplots(1, len(embed_labels))
    fig.set_figwidth(4 * len(embed_labels))
    co = 0
    for i, labels in enumerate(embed_labels):
        axes[i].scatter(embed_points[:, 0], embed_points[:, 1], c=labels, s=dot_size)
        axes[i].set_title(titles[i])
        if "K-means" in titles[i] and centers is not None and co == 0:
            axes[i].scatter(centers[:, 0], centers[:,1], c="none", s=dot_size, edgecolor="r")
            co += 1
    
    if main_title is not None:
        fig.suptitle(main_title)

    if save:
        if save_name is None:
            save_name = str(datetime.now())
        plt.savefig("savefiles/images/"+save_name+"_embeddings.png")
    
    plt.show()

