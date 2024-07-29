from utils import utils
from utils import utils_visual  # self

import numpy as np
import pandas as pd
import matplotlib as mlp
mlp.use('macosx')

def generate_data(seed=14):

    np.random.seed(seed)
    true_cluster = 1
    num_samples = 1000
    long = np.random.uniform(-5, 5, num_samples)
    lat = np.random.uniform(-5, 5, num_samples)
    y1 = np.random.uniform(1, 1, num_samples)  # placeholder
    x1 = np.random.uniform(-3, 3, num_samples)  # placeholder
    x2 = np.random.uniform(-3, 3, num_samples)  # placeholder
    data = pd.DataFrame({"true_clust": true_cluster, "long": long, "lat": lat, "x1": x1, "x2": x2, 'y1': y1})

    # subset the long/lat domain into mickey shape
    for i in range(len(data.index)):
        point1 = np.array((data['long'][i], data['lat'][i]))
        point2 = np.array((0, 0))
        dist1 = np.linalg.norm(point1 - point2)
        if (dist1 < 3.5 and dist1 > 2):
            data['true_clust'][i] = 2

    # generate Y's
    for i in range(len(data.index)):
        if data['true_clust'][i] == 1:
            data['y1'][i] = - data['x1'][i]**3
        elif data['true_clust'][i] == 2:
            data['y1'][i] = data['x1'][i]**3

    return data


def main_simul_1(tune=False, visualize=True, seeds=1):

    rep_scores = dict(gpsc_RI=[], gpsc_MI=[],
                      kmeans_RI=[], kmeans_MI=[],
                      spectral_RI=[], spectral_MI=[],
                      hier_RI=[], hier_MI=[],
                      dbscan_RI=[], dbscan_MI=[],
                      cmeans_RI=[], cmeans_MI=[],
                      spat_hier_RI=[], spat_hier_MI=[],
                      gdbscan_RI=[], gdbscan_MI=[],
                      gmm_RI=[], gmm_MI=[]
                      )

    # initialize placeholder for potential empty clusters
    empty_clusters = []
    # repeat over various seeds
    for seed in range(14, 14+seeds):

        ####################################################
        # generate data
        ####################################################

        np.random.seed(seed)
        data = generate_data(seed)

        ####################################################
        # apply GPSC
        ####################################################

        # randomize the initialization before GPSC
        data['new_clust'] = data['true_clust']
        data = utils.random_scramble(data, num_clust=2, save_random=True)

        from algorithms import gpsc_sklearn
        s_list = ['long', 'lat']
        x_list = ['x1', 'x2']
        y_list = ['y1']
        num_clusters = 2
        num_cycles =  20
        constant_lb = 1e-15
        constant_ub = 1e6 # 6 is best
        length_lb = 1e6   # 6 is best
        length_ub = 1e15
        lambda_ = 0
        # data_out = gpsc_sklearn.gp_cluster_isotropic_rbf_earlyStop(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub)
        try:
            data_out = gpsc_sklearn.gp_cluster_isotropic_rbf_lambda_earlyStop_RI_MI(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=lambda_, early_stop=0.90)
        except ValueError:
            print(ValueError)
            print(f"Empty Cluster on Random Seed: {seed}")
            empty_clusters.append(seed)
            continue
        ####################################################
        # apply traditional algorithms
        ####################################################

        kmeans_list = ['x1', 'x2', 'long', 'lat', "y1"]
        num_clusters = 2

        # 1. kmeans
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=num_clusters, random_state=0).fit(data[kmeans_list])

        # 2. spectral
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=5).fit(data[kmeans_list])
        if tune:
            utils.tune_parameters_spectral(data=data, kmeans_list=kmeans_list, num_clusters=num_clusters, print_thresh=0.1)
            #  SPECTRAL: max score = 0.166620591004722, n_neighbors= 5

        # 3. hierarchical
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(data[kmeans_list])

        # 4. DBSCAN
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=3, min_samples=3).fit(data[kmeans_list])
        if tune:
            utils.tune_parameters_dbscan(data, kmeans_list)
            #  DBSCAN: max=0.12436367206648, eps=3, min_samples=3

        ####################################################
        # apply traditional algorithms
        ####################################################

        # GDBSCAN
        import sys
        sys.path.append('../../')
        from algorithms.gdbscan import Points, GDBSCAN
        from algorithms.gdbscan_utils import w_card, Point
        import math

        if tune:
            utils.tune_parameters_gdbscan(data=data, cov_thresh=40, spatial_thresh=20, min_thresh=0, print_thresh=0.05)
            # GDBSCAN: score = 0.17197841143126616, covar_thresh = 3, spatial_thresh = 13, min_thresh = 0

        points = []
        for row in range(0, len(data.index)):
            points.append(Point(data['long'][row], data['lat'][row], data['x1'][row], data['x2'][row], data['y1'][row]))

        i = 3
        j = 13
        k = 0
        def n_pred(p1, p2):
            return all([
                math.sqrt((p1.val1 - p2.val1) ** 2 + (p1.val2 - p2.val2) ** 2 + (p1.val3 - p2.val3) ** 2) <= i,
                math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) <= j
                # math.sqrt((p1.val1 - p2.val1) ** 2 + (p1.val2 - p2.val2) ** 2 + (p1.val3 - p2.val3) ** 2 + (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) <= i
            ])

        clustered = GDBSCAN(Points(points), n_pred, k, w_card)

        results = pd.DataFrame()
        for cluster in clustered:
            for point in cluster:
                temp_df = pd.DataFrame(point.get_dict())
                results = results.append(temp_df, ignore_index=True)
        data_out = data_out.merge(results, how = 'inner')

        # spatialized hierarchical
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        if tune:
            utils.tune_parameters_spatial_hier(data=data, s_list=s_list, num_clusters=num_clusters, n_neighbors=11, print_threshold=0.1)
            # SPATIAL HIER: max = 0.025467435157629937, max_neigh=11, max_linkage=ward

        X = data[['long', 'lat']]
        knn_graph = kneighbors_graph(X, n_neighbors=11, include_self=False)
        model = AgglomerativeClustering(linkage='average', connectivity=knn_graph, n_clusters=num_clusters)
        model.fit(data[kmeans_list])
        data_out['spatial_hier'] = model.labels_

        # supervised fuzzy cmeans
        cmeans_init = []  # initialize cmeans for supervised learning
        data_max = max(data['y1'])
        data_min = min(data['y1'])
        for i in range(len(data.index)):
            if data['y1'][i] < 0:
                class_1 = 0.5 + (data['y1'][i] / data_min) / 2
                class_2 = 1 - class_1
                cmeans_init.append([class_1, class_2])
            elif data['y1'][i] > 0:
                class_2 = 0.5 + (data['y1'][i] / data_max) / 2
                class_1 = 1 - class_2
                cmeans_init.append([class_1, class_2])
            else:
                cmeans_init.append([0.5, 0.5])

        from skfuzzy.cluster import cmeans
        cntr, u, u0, d, jm, p, fpc = cmeans(data[kmeans_list].T, c=num_clusters, m=2, error=0.00001, maxiter=1000, init=np.array(cmeans_init).T)
        cluster_membership = np.argmax(u, axis=0)


        # final output


        from sklearn.mixture import GaussianMixture

        kmeans_list = ['x1', 'x2', 'long', 'lat', "y1"]
        gm = GaussianMixture(n_components=2, random_state=0).fit(data[kmeans_list])
        labels_gm = gm.predict(data[kmeans_list])
        data_out["GM"] = labels_gm



        labels_true = data_out['true_clust']
        labels_GPSC = data_out['new_clust']
        labels_fuzzyC = cluster_membership + 1
        labels_km = km.labels_
        labels_spectral = spectral.labels_ + 1
        labels_spat_hierarchical = data_out['spatial_hier']
        labels_gdbscan = data_out['gdbscan']
        labels_hierarchical = hierarchical.labels_ + 1
        labels_dbscan = dbscan.labels_ + 1

        data_out['fuzzy_cmeans'] = labels_fuzzyC
        data_out['spectral'] = labels_spectral
        data_out['km'] = labels_km + 1
        data_out['new_clust_hierarchical'] = labels_hierarchical
        data_out['new_clust_DBSCAN'] = labels_dbscan
        data_out['new_clust_GMM'] = labels_gm

        from sklearn import metrics

        print("rand index")
        print("GPSC - " + str(metrics.adjusted_rand_score(labels_true, labels_GPSC)))
        print("Fuzzy-C - " + str(metrics.adjusted_rand_score(labels_true, labels_fuzzyC)))
        print("Spectral - " + str(metrics.adjusted_rand_score(labels_true, labels_spectral)))
        print("Spatial-Hier - " + str(metrics.adjusted_rand_score(labels_true, labels_spat_hierarchical)))
        print("GDBSCAN - " + str(metrics.adjusted_rand_score(labels_true, labels_gdbscan)))
        print("KM - " + str(metrics.adjusted_rand_score(labels_true, km.labels_)))
        print("Hier - " + str(metrics.adjusted_rand_score(labels_true, labels_hierarchical)))
        print("DBSCAN - " + str(metrics.adjusted_rand_score(labels_true, labels_dbscan)))
        print("GMM - " + str(metrics.adjusted_rand_score(labels_true, labels_gm)))


        print()
        print("mutual info")
        print("GPSC - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_GPSC)))
        print("Fuzzy-C - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_fuzzyC)))
        print("Spectral - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_spectral)))
        print("Spatial-Hier - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_spat_hierarchical)))
        print("GDBSCAN - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_gdbscan)))
        print("KM - " + str(metrics.adjusted_mutual_info_score(labels_true, km.labels_)))
        print("Hier - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_hierarchical)))
        print("DBSCAN - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_dbscan)))
        print("GMM - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_gm)))

        # record scores for each seed
        rep_scores['gpsc_RI'].append(metrics.adjusted_rand_score(labels_true, labels_GPSC))
        rep_scores['gpsc_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_GPSC))

        rep_scores['kmeans_RI'].append(metrics.adjusted_rand_score(labels_true, km.labels_))
        rep_scores['kmeans_MI'].append(metrics.adjusted_mutual_info_score(labels_true, km.labels_))

        rep_scores['spectral_RI'].append(metrics.adjusted_rand_score(labels_true, labels_spectral))
        rep_scores['spectral_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_spectral))

        rep_scores['hier_RI'].append(metrics.adjusted_rand_score(labels_true, labels_hierarchical))
        rep_scores['hier_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_hierarchical))

        rep_scores['dbscan_RI'].append(metrics.adjusted_rand_score(labels_true, labels_dbscan))
        rep_scores['dbscan_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_dbscan))

        rep_scores['cmeans_RI'].append(metrics.adjusted_rand_score(labels_true, labels_fuzzyC))
        rep_scores['cmeans_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_fuzzyC))

        rep_scores['spat_hier_RI'].append(metrics.adjusted_rand_score(labels_true, labels_spat_hierarchical))
        rep_scores['spat_hier_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_spat_hierarchical))

        rep_scores['gdbscan_RI'].append(metrics.adjusted_rand_score(labels_true, labels_gdbscan))
        rep_scores['gdbscan_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_gdbscan))

        rep_scores['gmm_RI'].append(metrics.adjusted_rand_score(labels_true, labels_gm))
        rep_scores['gmm_MI'].append(metrics.adjusted_mutual_info_score(labels_true, labels_gm))

        # final plots
        ax_right = -76
        ax_left = 10

        data_out = utils.permute_clust(data=data_out, clustering='new_clust', clust1=1, clust2=2)
        data_out = utils.permute_clust(data=data_out, clustering='new_clust', clust1=1, clust2=2)


        if visualize:
            utils_visual.plot_spatial_data_matrix_simulation_2x4(data_out, x_list, y_list, s_list, ax_right=ax_right, ax_left=ax_left)
            utils_visual.plot_spatial_data_matrix_simulation_3x3_comparisons_2(data_out, x_list, y_list, ax_right=ax_right, ax_left=ax_left)
            utils_visual.psb_plot_simulation_comparisons_3x3_spatial(data_out, s_list, ax_right=ax_right, ax_left=ax_left, wspace=.2, hspace=.30, title_height=0.9)

    print(f"\n Final Scores and Averages:")
    # report average + std deviations
    for i in rep_scores.keys():
        print(f"{i} {np.mean(rep_scores[i])} {np.std(rep_scores[i])}")
    print(rep_scores)
    print(empty_clusters) # if any clusters returned empty

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    main_simul_1(tune=False, visualize=True, seeds=1)
