import numpy as np
import pandas as pd


def permute_clust(data, clustering, clust1, clust2):
    clust1 = int(clust1)
    clust2 = int(clust2)
    for i in range(len(data.index)):
        if data[clustering][i] == clust1:
            data[clustering][i] = clust2
        elif data[clustering][i] == clust2:
            data[clustering][i] = clust1

    return data


def random_scramble(data, num_clust, save_random = False):
    data['new_clust'] = np.random.randint(1, num_clust+1, len(data.index), dtype=int)
    if save_random:
        data['random_init'] = data['new_clust']

    return data


def tune_parameters_DEPRECATED(data, kmeans_list, num_clusters, thresh_spectral=0.25, thresh_dbscan=0.4):
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    spectral_scores = []
    dbscan_scores = []
    for i in range(1, 100, 1):
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=i).fit(
            data[kmeans_list])
        temp_score = metrics.adjusted_mutual_info_score(data['true_clust'], spectral.labels_ + 1)
        spectral_scores.append(temp_score)
        if (temp_score > thresh_spectral):
            print('n_neighbors= ' + str(i) + ', score =' + str(temp_score))
    print(str(max(spectral_scores)) + ' n_neighbors=' + str(spectral_scores.index(max(spectral_scores)) + 1))

    for i in range(1, 100, 1):
        for j in range(1, 300, 2):
            dbscan = DBSCAN(eps=i, min_samples=j).fit(data[kmeans_list])
            temp_score = metrics.adjusted_mutual_info_score(data['true_clust'], dbscan.labels_ + 1)
            dbscan_scores.append(temp_score)
            if (temp_score > thresh_dbscan):
                print('eps=' + str(i) + ', min_samples=' + str(j) + ', score =' + str(temp_score))
    print(max(dbscan_scores))


def tune_parameters_spectral(data, kmeans_list, num_clusters, print_thresh=1, monotone_report=True):
    from sklearn.cluster import SpectralClustering
    from sklearn import metrics

    spectral_scores = []
    max_score, max_neigh = 0, 0

    for i in range(1, 100, 1):
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=i).fit(data[kmeans_list])
        temp_score = metrics.adjusted_mutual_info_score(data['true_clust'], spectral.labels_ + 1)
        spectral_scores.append(temp_score)
        if temp_score > print_thresh:
            print('n_neighbors= ' + str(i) + ', score =' + str(temp_score))

        if (temp_score > max_score) and monotone_report:
            print('n_neighbors= ' + str(i) + ', score =' + str(temp_score))
            max_score = temp_score
            max_neigh = i

    print(f" SPECTRAL: max score = {max_score}, n_neighbors= {max_neigh}")


def tune_parameters_dbscan(data, kmeans_list, print_thresh=1, monotone_report=True):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    dbscan_scores = []
    max, max_eps, max_minSamples = 0, 0, 0
    for i in range(1, 100, 2):
        for j in range(1, 200, 2):
            dbscan = DBSCAN(eps=i, min_samples=j).fit(data[kmeans_list])
            temp_score = metrics.adjusted_mutual_info_score(data['true_clust'], dbscan.labels_ + 1)
            dbscan_scores.append(temp_score)
            if temp_score > print_thresh:
                print('eps=' + str(i) + ', min_samples=' + str(j) + ', score =' + str(temp_score))
            if temp_score > max and monotone_report:
                print('eps=' + str(i) + ', min_samples=' + str(j) + ', score =' + str(temp_score))
                max = temp_score
                max_eps = i
                max_minSamples = j

    print(f" DBSCAN: max={max}, eps={max_eps}, min_samples={max_minSamples}")


def tune_parameters_spatial_hier(data, s_list, num_clusters, print_threshold=1, n_neighbors=50, monotone_report=True):

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    from sklearn import metrics
    kmeans_list = ['long', 'lat', 'x1', 'x2', 'y1']
    X = data[s_list]
    spatial_hier_scores = []
    max_score = 0
    max_neigh = 0
    max_linkage = ''
    for i in range(1, n_neighbors, 2):
        knn_graph = kneighbors_graph(X, i, include_self=False)
        for linkage in ["average", "complete", "ward", "single"]:
            spatial_hier = AgglomerativeClustering(linkage=linkage, connectivity=knn_graph, n_clusters=num_clusters)
            spatial_hier.fit(data[kmeans_list])
            data['new_clust'] = spatial_hier.labels_
            temp_score = metrics.adjusted_mutual_info_score(data['true_clust'], spatial_hier.labels_ + 1)
            spatial_hier_scores.append(temp_score)
            if temp_score >= print_threshold:
                print(f"n_neighbors = {i}, linkage = {linkage}, score = {temp_score}")
            if temp_score > max_score and monotone_report:
                print(f"n_neighbors = {i}, linkage = {linkage}, score = {temp_score}")
                max_score = temp_score
                max_linkage = linkage
                max_neigh = i
    print(f"SPATIAL HIER: max = {max_score}, max_neigh={max_neigh}, max_linkage={max_linkage}")


def tune_parameters_gdbscan(data, cov_thresh, spatial_thresh, min_thresh, print_thresh=1, monotone_report=True, cov_step=2, cov_min=1, spatial_step=2, min_step=10):
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from algorithms.gdbscan import Points, GDBSCAN
    from algorithms.gdbscan_utils import w_card, Point
    import math
    from sklearn import metrics

    points = []
    for row in range(0, len(data.index)):
        points.append(Point(data['long'][row], data['lat'][row], data['x1'][row], data['x2'][row], data['y1'][row]))

    # tune gdbscan
    gdbscan_scores = []
    data_out = pd.DataFrame()
    max_score = 0
    max_i, max_j, max_k = 0, 0, 0
    for i in range(cov_min, cov_thresh+1, cov_step):
        for j in range(1, spatial_thresh+1, spatial_step):
            for k in range(0, min_thresh+1, min_step):
                def n_pred(p1, p2):
                    return all([
                        math.sqrt((p1.val1 - p2.val1) ** 2 + (p1.val2 - p2.val2) ** 2 + (p1.val3 - p2.val3) ** 2) <= i,
                        math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) <= j
                    ])

                clustered = GDBSCAN(Points(points), n_pred, k, w_card)
                results = pd.DataFrame()
                for cluster in clustered:
                    for point in cluster:
                        temp_df = pd.DataFrame(point.get_dict())
                        results = results.append(temp_df, ignore_index=True)
                data_out = data.merge(results, how='inner')

                temp_score = metrics.adjusted_mutual_info_score(data_out['true_clust'], data_out['gdbscan'] + 1)

                if (temp_score > print_thresh):
                    gdbscan_scores.append((i, j, k, temp_score))
                    print(f"covar_thresh = {i}, spatial_thresh = {j}, min_thresh = {k}, score = {temp_score}")

                if (temp_score > max_score) and monotone_report:
                    print(f"covar_thresh = {i}, spatial_thresh = {j}, min_thresh = {k}, score = {temp_score}")
                    max_score = temp_score
                    max_i, max_j, max_k = i, j, k

    print(f"GDBSCAN: score = {max_score}, covar_thresh = {max_i}, spatial_thresh = {max_j}, min_thresh = {max_k}")
    return data_out
