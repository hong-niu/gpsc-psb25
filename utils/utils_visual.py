import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
mlp.use('macosx')


def plot_spatial_data_matrix_simulation(data, x_list, y_list, space_list, ax_right, ax_left, out_path=None, label='new_clust'):

    # fig = plt.figure(figsize=(16, 14))
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(3, 3, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['true_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - True Cluster')

    ax = fig.add_subplot(3, 3, 2)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - True Cluster')

    ax = fig.add_subplot(3, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - True Cluster')

    ax = fig.add_subplot(3, 3, 4)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['random_init'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Random Cluster')

    ax = fig.add_subplot(3, 3, 5)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['random_init'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - Random Cluster')

    ax = fig.add_subplot(3, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['random_init'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - Random Cluster')

    ax = fig.add_subplot(3, 3, 7)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data[label])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Predicted Cluster')

    ax = fig.add_subplot(3, 3, 8)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - Predicted Cluster')

    ax = fig.add_subplot(3, 3, 9, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - Predicted Cluster')
    plt.subplots_adjust(wspace=0.2, hspace=0.325)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def plot_spatial_data_matrix_simulation_2x3(data, x_list, y_list, space_list, ax_right, ax_left, out_path=None, label='new_clust'):


    # fig = plt.figure(figsize=(16, 14))
    fig = plt.figure(figsize=(14, 8.5))
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['true_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - True Cluster')

    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - True Cluster')

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - True Cluster')

    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data[label])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Predicted Cluster')

    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - Predicted Cluster')

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - Predicted Cluster')
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt


def plot_spatial_data_matrix_simulation_2x4(data, x_list, y_list, space_list, ax_right, ax_left, out_path=None, label='new_clust'):


    # fig = plt.figure(figsize=(14, 8.5))
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.5), gridspec_kw={'width_ratios': [4, 4, 1.5, 4]})


    ax = plt.subplot(2, 4, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['true_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - True Cluster')

    ax = plt.subplot(2, 4, 2)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - True Cluster')

    ax = plt.subplot(2, 4, 3)
    # plt.ylim(-20, 15)
    y_pad = 0.1 * (float(np.max(data[y_list])) - float(np.min(data[y_list])))
    plt.ylim(float(np.min(data[y_list]))-y_pad, float(np.max(data[y_list])) + y_pad)
    plt.xlim(0.8, 1.2)
    cluster_1_y = data.loc[data['true_clust'] == 1]['y1']
    cluster_2_y = data.loc[data['true_clust'] == 2]['y1']
    cluster_3_y = data.loc[data['true_clust'] == 3]['y1']

    plot_y1 = np.ones(np.shape(cluster_1_y))  # Make all y values the same
    plot_y2 = np.ones(np.shape(cluster_2_y))
    plot_y3 = np.ones(np.shape(cluster_3_y))

    # ax.get_xaxis().set_visible(False)
    plt.title('Response - True Cluster')
    ax.set_xlabel(y_list[0])
    ax.set_xticklabels([])
    ax.plot(plot_y1, data.loc[data['true_clust'] == 1]['y1'], '_', ms=40, c='indigo', alpha=0.2)  # Plot a line at each location specified in a
    ax.plot(plot_y2, data.loc[data['true_clust'] == 2]['y1'], '_', ms=40, c='yellow', alpha=0.2)
    ax.plot(plot_y3, data.loc[data['true_clust'] == 3]['y1'], '_', ms=40, c='green', alpha=0.2)


    ax = plt.subplot(2, 4, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - True Cluster')

    ax = plt.subplot(2, 4, 5)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data[label])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Predicted Cluster')

    ax = plt.subplot(2, 4, 6)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - Predicted Cluster')

    ax = plt.subplot(2, 4, 7)
    plt.ylim(float(np.min(data[y_list]))-y_pad, float(np.max(data[y_list])) + y_pad)
    plt.xlim(0.8, 1.2)
    cluster_1_y = data.loc[data['new_clust'] == 1]['y1']
    cluster_2_y = data.loc[data['new_clust'] == 2]['y1']
    cluster_3_y = data.loc[data['new_clust'] == 3]['y1']

    plot_y1 = np.ones(np.shape(cluster_1_y))  # Make all y values the same
    plot_y2 = np.ones(np.shape(cluster_2_y))
    plot_y3 = np.ones(np.shape(cluster_3_y))

    # ax.get_xaxis().set_visible(False)
    ax.set_xlabel(y_list[0])
    ax.set_xticklabels([])
    plt.title('Response - Pred. Cluster')
    ax.plot(plot_y1, data.loc[data['new_clust'] == 1]['y1'], '_', ms=40, c='indigo', alpha=0.2)  # Plot a line at each location specified in a
    ax.plot(plot_y2, data.loc[data['new_clust'] == 2]['y1'], '_', ms=40, c='yellow', alpha=0.2)
    ax.plot(plot_y3, data.loc[data['new_clust'] == 3]['y1'], '_', ms=40, c='green', alpha=0.2)


    ax = plt.subplot(2, 4, 8, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - Predicted Cluster')


    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def plot_spatial_data_matrix_simulation_2x4_K3(data, x_list, y_list, space_list, ax_right, ax_left, out_path=None, label='new_clust'):


    # fig = plt.figure(figsize=(14, 8.5))
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.5), gridspec_kw={'width_ratios': [4, 4, 1.5, 4]})


    ax = plt.subplot(2, 4, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['true_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - True Cluster')

    ax = plt.subplot(2, 4, 2)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - True Cluster')

    ax = plt.subplot(2, 4, 3)
    plt.ylim(-500, 500)
    plt.xlim(0.5, 1.5)
    cluster_1_y = data.loc[data['true_clust'] == 1]['y1']
    cluster_2_y = data.loc[data['true_clust'] == 2]['y1']
    cluster_3_y = data.loc[data['true_clust'] == 3]['y1']
    plot_y1 = np.ones(np.shape(cluster_1_y))  # Make all y values the same
    plot_y2 = np.ones(np.shape(cluster_2_y))
    plot_y3 = np.ones(np.shape(cluster_3_y))
    # ax.get_xaxis().set_visible(False)
    plt.title('Response - True Cluster')
    ax.set_xlabel(y_list[0])
    ax.set_xticklabels([])
    ax.plot(plot_y1, data.loc[data['true_clust'] == 1]['y1'], '_', ms=40, c='indigo', alpha=0.05)  # Plot a line at each location specified in a
    ax.plot(plot_y3, data.loc[data['true_clust'] == 3]['y1'], '_', ms=40, c='yellow', alpha=0.2)
    ax.plot(plot_y2, data.loc[data['true_clust'] == 2]['y1'], '_', ms=40, c='green', alpha=0.15)


    ax = plt.subplot(2, 4, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - True Cluster')

    ax = plt.subplot(2, 4, 5)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data[label])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Predicted Cluster')

    ax = plt.subplot(2, 4, 6)
    ax.scatter(data[x_list[0]], data[x_list[1]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_title('Covariates - Predicted Cluster')

    ax = plt.subplot(2, 4, 7)
    plt.ylim(-500, 500)
    plt.xlim(0.5, 1.5)
    cluster_1_y = data.loc[data['new_clust'] == 1]['y1']
    cluster_2_y = data.loc[data['new_clust'] == 2]['y1']
    cluster_3_y = data.loc[data['new_clust'] == 3]['y1']
    plot_y1 = np.ones(np.shape(cluster_1_y))  # Make all y values the same
    plot_y2 = np.ones(np.shape(cluster_2_y))
    plot_y3 = np.ones(np.shape(cluster_3_y))
    # ax.get_xaxis().set_visible(False)
    ax.set_xlabel(y_list[0])
    ax.set_xticklabels([])
    plt.title('Response - Pred. Cluster')
    ax.plot(plot_y1, data.loc[data['new_clust'] == 1]['y1'], '_', ms=40, c='indigo', alpha=0.05)  # Plot a line at each location specified in a
    ax.plot(plot_y3, data.loc[data['new_clust'] == 3]['y1'], '_', ms=40, c='yellow', alpha=0.2)
    ax.plot(plot_y2, data.loc[data['new_clust'] == 2]['y1'], '_', ms=40, c='green', alpha=0.15)



    ax = plt.subplot(2, 4, 8, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data[label])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Cov./Response - Predicted Cluster')


    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def plot_spatial_data_matrix_simulation_comparisons(data, x_list, y_list, ax_right, ax_left, out_path=None):

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('True Cluster')

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GPSC')

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_km'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('K-Means')

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_spectral'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spectral')

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_hierarchical'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Hierarchical')

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_DBSCAN'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('DBSCAN')
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def plot_spatial_data_matrix_simulation_comparisons_2(data, x_list, y_list, ax_right, ax_left, out_path=None):

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('True Cluster', y=0.85)

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GPSC', y=0.85)

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Super-Fuzzy-Cmeans', y=0.85)

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spectral'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spectral', y=0.85)

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spatial_hier'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spatial-Hierarchical', y=0.85)

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['gdbscan'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GDBSCAN', y=0.85)

    #f#ig=ax.get_figure()
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.95)

    plt.subplots_adjust(wspace=0.1, hspace=-0.07)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt


def tmlr_plot_simulation_comparisons_2x4_clean(data, x_list, y_list, ax_right, ax_left, wspace=0, hspace=0, title_height=0.85, out_path=None):

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(2, 4, 1, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    # ax.set_xlabel(x_list[0])
    # ax.set_ylabel(x_list[1])
    # ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('True Cluster', y=title_height)

    ax = fig.add_subplot(2, 4, 2, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['km'])
    # ax.set_xlabel(x_list[0])
    # ax.set_ylabel(x_list[1])
    # ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('K-Means', y=title_height)

    ax = fig.add_subplot(2, 4, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_hierarchical'])
    # ax.set_xlabel(x_list[0])
    # ax.set_ylabel(x_list[1])
    # ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Hierarchical', y=title_height)

    ax = fig.add_subplot(2, 4, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_DBSCAN'])
    # ax.set_xlabel(x_list[0])
    # ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('DBSCAN', y=title_height)

    ax = fig.add_subplot(2, 4, 5, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spectral'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_title('Spectral', y=title_height)

    ax = fig.add_subplot(2, 4, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_title('Super. Fuzzy C-Means', y=title_height)

    ax = fig.add_subplot(2, 4, 7, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spatial_hier'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_zticklabels([])
    ax.set_title('Spatial Hierarchical', y=title_height)

    ax = fig.add_subplot(2, 4, 8, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['gdbscan'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GDBSCAN', y=title_height)

    #fig=ax.get_figure()
    # fig.tight_layout()
    #fig.subplots_adjust(top=0.95)

    # plt.subplots_adjust(wspace=0.225, hspace=-0.425) # for all axes
    # plt.subplots_adjust(wspace=0.075, hspace=-0.575) # for clean
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def tmlr_plot_simulation_comparisons_2x4_spatial_clean(data, space_list, ax_right, ax_left, wspace=0, hspace=0, title_height=0.85, out_path=None):

    fig = plt.figure(figsize=(15, 7))

    ax = fig.add_subplot(2, 4, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['true_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - True Cluster')

    ax = fig.add_subplot(2, 4, 2)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['km'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - K-means')

    ax = fig.add_subplot(2, 4, 3)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust_hierarchical'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Hierarchical')

    ax = fig.add_subplot(2, 4, 4)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust_DBSCAN'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - DBSCAN')

    ax = fig.add_subplot(2, 4, 5)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['spectral'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Spectral')

    ax = fig.add_subplot(2, 4, 6)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Fuzzy C-Means')

    ax = fig.add_subplot(2, 4, 7)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['spatial_hier'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - Spatial Hierarchical')

    ax = fig.add_subplot(2, 4, 8)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['gdbscan'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spatial - GDBSCAN')

    #fig=ax.get_figure()
    # fig.tight_layout()
    #fig.subplots_adjust(top=0.95)

    # plt.subplots_adjust(wspace=0.225, hspace=-0.425) # for all axes
    # plt.subplots_adjust(wspace=0.075, hspace=-0.575) # for clean
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def tmlr_plot_simulation_comparisons_3x3_clean(data, x_list, y_list, ax_right, ax_left, wspace=0, hspace=0, title_height=0.85, out_path=None):

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(3, 3, 1, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['true_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('True Cluster', y=title_height)

    ax = fig.add_subplot(3, 3, 2, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GPSC', y=title_height)

    ax = fig.add_subplot(3, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spectral'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spectral', y=title_height)

    ax = fig.add_subplot(3, 3, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['km'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('K-Means', y=title_height)

    ax = fig.add_subplot(3, 3, 5, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_hierarchical'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Hierarchical', y=title_height)

    ax = fig.add_subplot(3, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_DBSCAN'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('DBSCAN', y=title_height)

    ax = fig.add_subplot(3, 3, 7, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Super. Fuzzy C-Means', y=title_height)

    ax = fig.add_subplot(3, 3, 8, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spatial_hier'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spatial Hierarchical', y=title_height)

    ax = fig.add_subplot(3, 3, 9, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['gdbscan'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GDBSCAN', y=title_height)

    # fig=ax.get_figure()
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)

    # plt.subplots_adjust(wspace=0.225, hspace=-0.425) # for all axes
    # plt.subplots_adjust(wspace=0.075, hspace=-0.575) # for clean
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt


# for tmlr - include GMM
def psb_plot_simulation_comparisons_3x3_spatial(data, space_list, ax_right, ax_left, wspace=0, hspace=0, title_height=0.85, out_path=None):

    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot(3, 3, 1)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('GPSC')

    ax = fig.add_subplot(3, 3, 2)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['GM'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('GMM')

    ax = fig.add_subplot(3, 3, 3)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['spectral'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spectral')

    ax = fig.add_subplot(3, 3, 4)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust_hierarchical'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Hierarchical')

    ax = fig.add_subplot(3, 3, 5)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['spatial_hier'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Spat. Hierarchical')

    ax = fig.add_subplot(3, 3, 6)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['km'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('K-means')

    ax = fig.add_subplot(3, 3, 7)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('Fuzzy C-Means')

    ax = fig.add_subplot(3, 3, 8)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['new_clust_DBSCAN'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('DBSCAN')

    ax = fig.add_subplot(3, 3, 9)
    ax.scatter(data[space_list[0]], data[space_list[1]], c=data['gdbscan'])
    ax.set_xlabel(space_list[0])
    ax.set_ylabel(space_list[1])
    ax.set_title('GDBSCAN')

    #fig=ax.get_figure()
    # fig.tight_layout()
    #fig.subplots_adjust(top=0.95)

    # plt.subplots_adjust(wspace=0.225, hspace=-0.425) # for all axes
    # plt.subplots_adjust(wspace=0.075, hspace=-0.575) # for clean
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt

def plot_spatial_data_matrix_simulation_3x3_comparisons_2(data, x_list, y_list, ax_right, ax_left, out_path=None, wspace=-0.25, hspace=-0.06):

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(3, 3, 1, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GPSC', y=0.85)

    ax = fig.add_subplot(3, 3, 2, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['GM'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GMM', y=0.85)

    ax = fig.add_subplot(3, 3, 3, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spectral'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spectral', y=0.85)

    ax = fig.add_subplot(3, 3, 4, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_hierarchical'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Hierarchical', y=0.85)

    ax = fig.add_subplot(3, 3, 5, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['spatial_hier'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Spatial-Hierarchical', y=0.85)

    ax = fig.add_subplot(3, 3, 6, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['km'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('K-means', y=0.85)

    ax = fig.add_subplot(3, 3, 7, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['fuzzy_cmeans'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('Super-Fuzzy-Cmeans', y=0.85)

    ax = fig.add_subplot(3, 3, 8, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['new_clust_DBSCAN'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('DBSCAN', y=0.85)

    ax = fig.add_subplot(3, 3, 9, projection='3d')
    ax.view_init(ax_left, ax_right)
    ax.scatter(data[x_list[0]], data[x_list[1]], data[y_list[0]], c=data['gdbscan'])
    ax.set_xlabel(x_list[0])
    ax.set_ylabel(x_list[1])
    ax.set_zlabel(y_list[0])
    ax.set_title('GDBSCAN', y=0.85)

    #f#ig=ax.get_figure()
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.95)

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    plt.show()
    return plt
