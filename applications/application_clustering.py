from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
import numpy as np
import math


# no lambda penalty
def gp_cluster_direct(data, x_List, y_List, numClusters, numCycles, constant_LB, constant_UB, length_LB, length_UB):

    kernel = ConstantKernel(constant_value_bounds=(constant_LB, constant_UB)) * RBF(length_scale_bounds=(length_LB, length_UB))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    new_data_xi = data[x_List]

    for cycle in range(1, numCycles+1):
        for i in range(1, numClusters+1):
            x_train_i = data.loc[data['new_clust'] == i][x_List] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_List]

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(gp_fit.kernel_)
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data_xi, return_std=False)

        for j in range(0,len(data.index)):
            diff_list = []
            for i in range(1, numClusters+1):
                diff_i = abs(data[y_List[0]][j] - data['gp'+str(i)+'_mean_pred'][j])**2 # prediction error from GP
                diff_list.append(diff_i)
            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

    return data

def k_means_spatial(data, numCycles, lambda_):

    for i in range(1, numCycles):
        cluster_1_long = data.loc[data['new_clust'] == 1]['LONGITUDE'].to_numpy()  # loc to find =2, ['long']
        cluster_1_lat = data.loc[data['new_clust'] == 1]['LATITUDE'].to_numpy()

        cluster_2_long = data.loc[data['new_clust'] == 2]['LONGITUDE'].to_numpy()
        cluster_2_lat = data.loc[data['new_clust'] == 2]['LATITUDE'].to_numpy()

        cluster_3_long = data.loc[data['new_clust'] == 3]['LONGITUDE'].to_numpy()
        cluster_3_lat = data.loc[data['new_clust'] == 3]['LATITUDE'].to_numpy()

        # reclassify clusters
        center_1_long = sum(cluster_1_long) / len(cluster_1_long)
        center_1_lat = sum(cluster_1_lat) / len(cluster_1_lat)

        center_2_long = sum(cluster_2_long) / len(cluster_2_long)
        center_2_lat = sum(cluster_2_lat) / len(cluster_2_lat)

        center_3_long = sum(cluster_3_long) / len(cluster_3_long)
        center_3_lat = sum(cluster_3_lat) / len(cluster_3_lat)

        for i in range(1, len(data.index)):
            diff_1 = lambda_ * math.sqrt(
                abs(data['LONGITUDE'][i] - center_1_long) ** 2 + abs(data['LATITUDE'][i] - center_1_lat) ** 2)
            diff_2 = lambda_ * math.sqrt(
                abs(data['LONGITUDE'][i] - center_2_long) ** 2 + abs(data['LATITUDE'][i] - center_2_lat) ** 2)
            diff_3 = lambda_ * math.sqrt(
                abs(data['LONGITUDE'][i] - center_3_long) ** 2 + abs(data['LATITUDE'][i] - center_3_lat) ** 2)
            diff_list = [diff_1, diff_2, diff_3]
            data.iloc[i, data.columns.get_loc('new_clust')] = diff_list.index(
                min(diff_list)) + 1  # update the new_clusts

    return data

