import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mlp
mlp.use('macosx')


def plot_spatial_data_matrix(data, x_list, y_list, space_list):

    fig = plt.figure(figsize=plt.figaspect(0.45))

    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust'])
    ax.set_title('Spatial - Predicted Cluster')

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['new_clust'])
    ax.set_title('Covariates - Predicted Cluster')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust'])
    ax.set_title('Covariate/Response - Predicted Cluster')
    plt.show()

    return plt



def plot_spatial_data_map(data, space_list):


    ax = plt.axes()
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data["new_clust"])
    plt.legend()
    plt.xlabel(space_list[0])
    plt.ylabel(space_list[1])
    plt.show()
    return plt

