import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import FuncFormatter


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

