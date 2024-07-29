import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.use('macosx')
from utils import utils
from utils import utils_appl
from tabulate import tabulate
import application_clustering
import pandas as pd



def real_multi_cov_GPSC():
    data = pd.read_excel("/XXXX/", index_col=0)

    x_list = [ "LONGITUDE", "LATITUDE", "PRFL_M", "PRFL_F",  "LS_HS", "SINGLE", "HSHLDR_F", "NHBLK", "PA", "POV", "NO_VHCL", "RENT", "CROWD", "UNMPLYD", "PHONE", "ACET", "BENZENE", "BUTA", "CARBON", "ETHYL", "FORM", "HEXANE", "LEAD", "MANG", "MERC", "METH", "METHYL", "NICK", "TOLUENE", "XYLENE"]
    y_list = ["MLCJOINT2"]
    kmeans_list = [ "LONGITUDE", "LATITUDE", "PRFL_M", "PRFL_F",  "LS_HS", "SINGLE", "HSHLDR_F", "NHBLK", "PA", "POV", "NO_VHCL", "RENT", "CROWD", "UNMPLYD", "PHONE", "ACET", "BENZENE", "BUTA", "CARBON", "ETHYL", "FORM", "HEXANE", "LEAD", "MANG", "MERC", "METH", "METHYL", "NICK", "TOLUENE", "XYLENE", "MLCJOINT2"]
    space_list = ["LONGITUDE", "LATITUDE"]
    num_clusters = 3
    data = data.dropna().reset_index(drop=True)
    data = utils.random_scramble(data, num_clusters)

    # check
    print(tabulate(data, headers='keys'))
    utils_appl.plot_spatial_data_matrix(data, x_list, y_list, space_list)
    utils_appl.plot_spatial_data_map(data, space_list)


    data = application_clustering.k_means_spatial(data, 10, lambda_=1)

    # note that x_list contains the spatial longitude and latitude variables
    # so that x_list = [s,x] put into one vector
    data = application_clustering.gp_cluster_direct(data, x_List=x_list, y_List=y_list, numClusters=num_clusters, numCycles=7, lambda_=0, constant_LB=1e-15, constant_UB=1e3, length_LB=1e6, length_UB=1e15)
    utils_appl.plot_spatial_data_map(data, space_list)
    return data

def real_multi_cov_kmeans(num_clust):
    data = pd.read_excel("XXXX", index_col=0)

    x_list = ["LONGITUDE", "LATITUDE","PRFL_M", "PRFL_F",  "LS_HS", "SINGLE", "HSHLDR_F", "NHBLK", "PA", "POV", "NO_VHCL", "RENT", "CROWD", "UNMPLYD", "PHONE", "ACET", "BENZENE", "BUTA", "CARBON", "ETHYL", "FORM", "HEXANE", "LEAD", "MANG", "MERC", "METH", "METHYL", "NICK", "TOLUENE", "XYLENE", "MLCJOINT2"]
    space_list = ["LONGITUDE", "LATITUDE"]

    num_clusters = num_clust
    data = data.dropna().reset_index(drop=True)
    data = utils.random_scramble(data, num_clusters)
    x_train = data[x_list]

    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=3, random_state=0).fit(data[x_train])

    ax = plt.axes()
    ax.scatter(data[space_list[0]], data[space_list[1]], c=gm.labels_)
    plt.show()
    plt.clf()
    data['new_clust'] = gm.labels_ + 1
    return data

def run_tract_kmeans_psb():
    # K-means Tract Level Results
    data = real_multi_cov_kmeans(num_clust=3)
    cluster_filename = "./derived_data/clusters_tract_k3_kmeans.xlsx"
    data.to_excel(cluster_filename)
    data = pd.read_excel(cluster_filename)

    title = 'KMeans - Tract Level - L=3'
    out = "./figures/tract_experiments/tract-kmeans.png"
    census_tract_plotting.plot_tracts(cluster_filename, title, out_path=out)

    x_limits = [-81.5, -78]
    y_limits = [34.6, 36.4]
    out = "./figures/tract_experiments/tract-kmeans-central.png"
    title = 'KMeans - Central'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-78.6, -77.2]
    y_limits = [33.8, 34.6]
    out = "./figures/tract_experiments/tract-kmeans-wilm.png"
    title = 'KMeans - Southeastern'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-83.4, -82.0]
    y_limits = [34.9, 35.7]
    out = "./figures/tract_experiments/tract-kmeans-ashe.png"
    title = 'KMeans - Western'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-79.7, -78.3]
    y_limits = [35.5, 36.3]
    out = "./figures/tract_experiments/tract-kmeans-central-zoom.png"
    title = 'KMeans - Central'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)




def run_tract_GPSC_psb():
    # GPSC Tract Level
    #data = real_test_multi_cov_GPSC()
    cluster_filename = "derived_data/clusters_tract_k3_GPSC.xlsx"
    data = pd.read_excel(cluster_filename)
    # data = utils.permute_clust(data, 'new_clust', 2, 3)
    # data = utils.permute_clust(data, 'new_clust', 1, 2)
    # data.to_excel(cluster_filename)
    title = 'GPSC - Tract Level - L=3'
    out = "./figures/tract_experiments/tract-GPSC.png"
    census_tract_plotting.plot_tracts(cluster_filename, title, out_path=out)

    x_limits=[-81.5, -78]
    y_limits=[34.6, 36.4]
    out = "./figures/tract_experiments/tract-GPSC-central.png"
    title = 'GPSC - Central'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-78.6, -77.2]
    y_limits = [33.8, 34.6]
    out = "./figures/tract_experiments/tract-GPSC-wilm.png"
    title = 'GPSC - Southeastern'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-83.4, -82.0]
    y_limits = [34.9, 35.7]
    out = "./figures/tract_experiments/tract-GPSC-ashe.png"
    title = 'GPSC - Western'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)

    x_limits = [-79.7, -78.3]
    y_limits = [35.5, 36.3]
    out = "./figures/tract_experiments/tract-GPSC-central-zoom.png"
    title = 'GPSC - Central'
    census_tract_plotting.plot_tracts_subfig(cluster_filename, title, out_path=out, x_limits=x_limits, y_limits=y_limits)


if __name__ == '__main__':

    import census_tract_plotting
    np.random.seed(52)

    # run tract level analyses

    run_tract_kmeans_psb()
    # run_tract_GPSC_psb()

    # compute agreement between clusterings

    cluster_filename_1 = "./derived_data/clusters_tract_k3_kmeans.xlsx"
    cluster_filename_2 = "./derived_data/clusters_tract_k3_GPSC.xlsx"

    data1 = pd.read_excel(cluster_filename_1)
    data2 = pd.read_excel(cluster_filename_2)

    from sklearn import metrics

    print(metrics.adjusted_rand_score(data1['new_clust'], data2['new_clust']))
    print(metrics.adjusted_mutual_info_score(data1['new_clust'], data2['new_clust']))

