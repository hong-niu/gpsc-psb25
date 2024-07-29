from utils import utils
from utils import utils_visual  # self
import numpy as np
import pandas as pd


def generate_data(seed=14):

    np.random.seed(seed)
    true_cluster = 1
    num_samples = 1000
    long = np.random.uniform(-5, 5, num_samples)
    lat = np.random.uniform(-5, 5, num_samples)
    y1 = np.random.uniform(1, 1, num_samples)  # placeholder
    x1 = np.random.uniform(-6, 6, num_samples)  # placeholder
    x2 = np.random.uniform(-2, 4, num_samples)  # placeholder
    data = pd.DataFrame({"true_clust": true_cluster, "long": long, "lat": lat, "x1": x1, "x2": x2, 'y1': y1})

    # subset the long/lat domain into sun and moon shape
    point2 = np.array((-2.2, 2.2))
    point3 = np.array((1.8, -1.8))
    point4 = np.array((1, -1))
    for i in range(len(data.index)):
        point1 = np.array((data['long'][i], data['lat'][i]))
        dist1 = np.linalg.norm(point1 - point2)
        if dist1 < 2.5:
            data['true_clust'][i] = 2
        dist2 = np.linalg.norm(point1 - point3)
        dist3 = np.linalg.norm(point1 - point4)
        if dist2 < 3 and dist3 > 2:
            data['true_clust'][i] = 3

    # generate Y's
    for i in range(len(data.index)):
        y1 = data['y1'][i]
        x1 = data['x1'][i]
        x2 = data['x2'][i]
        s1 = data['long'][i]
        s2 = data['lat'][i]
        if data['true_clust'][i] == 1:
            data['y1'][i] = 40*x1**2 - 400  + np.random.normal(0, 2)
        elif data['true_clust'][i] == 2:
            data['y1'][i] = -(x1-8)**3  + np.random.normal(0, 2)
        elif data['true_clust'][i] == 3:
            data['y1'][i] = (x1+8) **3 - 20 + np.random.normal(0, 2)
    return data

def main_simul_2(tune = False, visualize=True, seeds=1):

    # initialize dictionary for scores
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

    # loop through seeds
    for seed in range(14, 14 + seeds):

        ####################################################
        # generate data
        ####################################################

        np.random.seed(seed)
        print(f"Random Seed is: {seed}")

        data = generate_data(seed)

        ####################################################
        # apply GPSC
        ####################################################

        # randomize the initialization before GPSC
        data['new_clust'] = data['true_clust'] # initialize column, will be randomized
        data = utils.random_scramble(data, num_clust=6, save_random=True)

        # apply GPSC
        from algorithms import gpsc_sklearn
        s_list = ['long', 'lat']
        x_list = ['x1', 'x2']
        y_list = ['y1']
        num_clusters = 6
        num_cycles = 40
        constant_lb = 1e-15
        constant_ub = 1e3 # 3
        length_lb = 1e4 # 4
        length_ub = 1e15
        # lambda_= 30
        lambda_= 75 # use 75 for noise = 200, 50 for noise = 1
        try:
            data = gpsc_sklearn.gp_cluster_isotropic_rbf_lambda_earlyStop_RI_MI(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=lambda_, early_stop=0.90)
            # data = gpsc_sklearn.gp_cluster_isotropic_rbf_lambda_earlyStop_RI_MI_nugget(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=lambda_, early_stop=0.90)

        except ValueError:
            print(ValueError)
            print(f"Empty Cluster on Random Seed: {seed}")
            empty_clusters.append(seed)
            continue

        ####################################################
        # apply traditional algorithms
        ####################################################

        # 1. k-means
        kmeans_list = ['x1', 'x2', 'long', 'lat', "y1"]

        num_clusters = 6
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=num_clusters, random_state=0).fit(data[kmeans_list])

        # 2. spectral
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=12).fit(data[kmeans_list])
        if tune:
            utils.tune_parameters_spectral(data=data, kmeans_list=kmeans_list, num_clusters=num_clusters, print_thresh=0.1)
            #  SPECTRAL: max score = 0.16929329289596715, n_neighbors= 12

        # 3. hierarchical
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(data[kmeans_list])

        # 5. DBSCAN
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=41, min_samples=27).fit(data[kmeans_list])
        if tune:
            utils.tune_parameters_dbscan(data, kmeans_list)
            #  DBSCAN: max=0.25481766628737507, eps=41, min_samples=27
        ####################################################
        # apply spatial algorithms
        ####################################################

        # 6. GDBSCAN

        if tune:
            utils.tune_parameters_gdbscan(data=data, cov_thresh=800, spatial_thresh=9, cov_step=25, cov_min=450, min_thresh=0, print_thresh=0.25)
            # GDBSCAN: score = 0.3552927075513507, covar_thresh = 675, spatial_thresh = 5, min_thresh = 0

        import sys
        sys.path.append('../../')
        from algorithms.gdbscan import Points, GDBSCAN
        from algorithms.gdbscan_utils import w_card, Point
        import math

        points = []
        for row in range(0, len(data.index)):
            points.append(Point(data['long'][row], data['lat'][row], data['x1'][row], data['x2'][row], data['y1'][row]))

        cov = 675
        spatial = 5
        min_gdb = 0

        def n_pred(p1, p2):
            return all([
                math.sqrt((p1.val1 - p2.val1) ** 2 + (p1.val2 - p2.val2) ** 2 + (p1.val3 - p2.val3) ** 2) <= cov,
                math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) <= spatial
            ])

        clustered = GDBSCAN(Points(points), n_pred, min_gdb, w_card)

        results = pd.DataFrame()
        for cluster in clustered:
            for point in cluster:
                temp_df = pd.DataFrame(point.get_dict())
                results = results.append(temp_df, ignore_index=True)
        data_out = data.merge(results, how='inner')

        # 7. spatialized hierarchical

        if tune:
            utils.tune_parameters_spatial_hier(data=data, s_list=s_list, num_clusters=num_clusters, n_neighbors=75, print_threshold=0.62)
            # SPATIAL HIER: max = 0.3145854644883533, max_neigh=9, max_linkage=ward

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        X = data[['long', 'lat']]
        knn_graph = kneighbors_graph(X, 9, include_self=False)
        model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=num_clusters)
        model.fit(data[kmeans_list])
        data_out['spatial_hier'] = model.labels_

        # 8. supervised fuzzy cmeans

        cmeans_init = []  # initialize cmeans for supervised learning
        for i in range(len(data.index)):
            if data['y1'][i] < -150:
                class_1 = 0.6
                class_2 = 0.3
                class_3 = 0.1
                cmeans_init.append([class_1, class_2, class_3])
            elif data['y1'][i] > -150 and data['y1'][i] < 150 :
                class_1 = 0.2
                class_2 = 0.6
                class_3 = 0.2
                cmeans_init.append([class_1, class_2, class_3])
            else:
                class_1 = 0.1
                class_2 = 0.3
                class_3 = 0.6
                cmeans_init.append([class_1, class_2, class_3])

        from skfuzzy.cluster import cmeans
        cntr, u, u0, d, jm, p, fpc = cmeans(data[kmeans_list].T, c=num_clusters, m=2, error=0.00001, maxiter=1000, init=np.array(cmeans_init).T)# np.array(cmeans_init).T)
        cluster_membership = np.argmax(u, axis=0)

        # GMM
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=num_clusters, random_state=0).fit(data[kmeans_list])
        labels_gm = gm.predict(data[kmeans_list])
        data_out["GM"] = labels_gm

        ####################################################
        # final output
        ####################################################

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
        print("GM - " + str(metrics.adjusted_rand_score(labels_true, labels_gm)))

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
        print("GM - " + str(metrics.adjusted_mutual_info_score(labels_true, labels_gm)))
        print()

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
        ax_right = -99
        ax_left = 2

        data_out = utils.permute_clust(data=data_out, clustering='new_clust', clust1=1, clust2=3)
        data_out = utils.permute_clust(data=data_out, clustering='spatial_hier', clust1=1, clust2=3)
        data_out = utils.permute_clust(data=data_out, clustering='fuzzy_cmeans', clust1=1, clust2=2)
        data_out = utils.permute_clust(data=data_out, clustering='fuzzy_cmeans', clust1=1, clust2=2)


        if visualize:
            utils_visual.plot_spatial_data_matrix_simulation_2x4(data_out, x_list, y_list, s_list, ax_right=ax_right, ax_left=ax_left)
            utils_visual.plot_spatial_data_matrix_simulation_3x3_comparisons_2(data_out, x_list, y_list, ax_right=ax_right, ax_left=ax_left)
            utils_visual.icml_plot_simulation_comparisons_3x3_spatial(data_out, s_list, ax_right=ax_right, ax_left=ax_left, wspace=.2, hspace=.30, title_height=0.9)

    # final average scores outputs
    print(f"Final Scores and Averages")
    for i in rep_scores.keys():
        print(f"{i} {np.mean(rep_scores[i])} {np.std(rep_scores[i])}")
    print(rep_scores)
    print(empty_clusters) # if any clusters returned empty


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    main_simul_2(tune=False, visualize=True, seeds=1)

    print("k=6, noise=2")
