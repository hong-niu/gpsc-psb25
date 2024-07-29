from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel

from sklearn import metrics
import numpy as np


def gp_cluster_isotropic_rbf(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub):

    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(gp_fit.kernel_)
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j])
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

    return data

def gp_cluster_isotropic_rbf_earlyStop(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub):

    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]
    data['new_clust_prev'] = data['new_clust']

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(f"GPSC Round: {cycle}, Clust_{i} Kernel = {gp_fit.kernel_}")
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j])
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

        # stop if the clusters stabilize
        mi_update = metrics.adjusted_mutual_info_score(data['new_clust_prev'], data['new_clust'])
        print(f" MI_update = {mi_update} for cycle: {cycle}")
        # if mi_update > 0.95:
        if mi_update > 0.99:
            print(f"Stopped early on cycle: {cycle}")
            return data

        # update previous
        data['new_clust_prev'] = data['new_clust']

    return data


def gp_cluster_isotropic_rbf_lambda(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=0):

    import math

    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            if len(x_train_i) == 0:
                raise ValueError('Empty Cluster Detected')

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(f"GPSC Round: {cycle}, Clust_{i} Kernel = {gp_fit.kernel_}")
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                cluster_i_s1 = data.loc[data['new_clust'] == i]['long'].to_numpy()  # loc to find =2, ['x1'])
                cluster_i_s2 = data.loc[data['new_clust'] == i]['lat'].to_numpy()
                if len(cluster_i_s1) == 0:
                    raise ValueError('Empty Cluster Detected')
                center_i_s1 = sum(cluster_i_s1) / len(cluster_i_s1)
                center_i_s2 = sum(cluster_i_s2) / len(cluster_i_s2)

                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j]) + (lambda_ * math.sqrt(abs( data['long'][j]- center_i_s1)**2 + abs( data['lat'][j]- center_i_s2)**2))
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

    return data


def gp_cluster_isotropic_rbf_lambda_earlyStop(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=0):
    import math

    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]
    data['new_clust_prev'] = data['new_clust']

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            if len(x_train_i) == 0:
                raise ValueError('Empty Cluster Detected')

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(f"GPSC Round: {cycle}, Clust_{i} Kernel = {gp_fit.kernel_}")
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                cluster_i_s1 = data.loc[data['new_clust'] == i]['long'].to_numpy()  # loc to find =2, ['x1'])
                cluster_i_s2 = data.loc[data['new_clust'] == i]['lat'].to_numpy()
                if len(cluster_i_s1) == 0:
                    raise ValueError('Empty Cluster Detected')
                center_i_s1 = sum(cluster_i_s1) / len(cluster_i_s1)
                center_i_s2 = sum(cluster_i_s2) / len(cluster_i_s2)

                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j]) + (lambda_ * math.sqrt(abs( data['long'][j]- center_i_s1)**2 + abs( data['lat'][j]- center_i_s2)**2))
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

        MI_update = metrics.adjusted_mutual_info_score(data['new_clust_prev'], data['new_clust'])
        print(f" MI_update = {MI_update} for cycle: {cycle}")
        if MI_update > 0.90:
            print(f"Stopped early on cycle: {cycle}")
            return data
        # update previous
        data['new_clust_prev'] = data['new_clust']

    return data


def gp_cluster_isotropic_rbf_lambda_earlyStop_RI_MI(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=0, early_stop=0.9):
    import math

    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]
    data['new_clust_prev'] = data['new_clust']

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            if len(x_train_i) == 0:
                raise ValueError('Empty Cluster Detected')

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(f"GPSC Round: {cycle}, Clust_{i} Kernel = {gp_fit.kernel_}")
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        # calculate the centers
        center_i_s1 = []
        center_i_s2 = []
        for clust in range(1, num_clusters + 1):
            cluster_i_s1 = data.loc[data['new_clust'] == clust]['long'].to_numpy()  # loc to find =2, ['x1'])
            cluster_i_s2 = data.loc[data['new_clust'] == clust]['lat'].to_numpy()
            if len(cluster_i_s1) == 0:
                raise ValueError('Empty Cluster Detected')
            center_i_s1.append(sum(cluster_i_s1) / len(cluster_i_s1))
            center_i_s2.append(sum(cluster_i_s2) / len(cluster_i_s2))

        # update clusters
        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j]) + (lambda_ * math.sqrt(abs( data['long'][j]- center_i_s1[i-1])**2 + abs( data['lat'][j]- center_i_s2[i-1])**2))
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

        RI_update = metrics.adjusted_rand_score(data['new_clust_prev'], data['new_clust'])
        MI_update = metrics.adjusted_mutual_info_score(data['new_clust_prev'], data['new_clust'])
        print(f" RI_update = {round(RI_update, 3)}, MI_update = {round(MI_update, 3)}, for cycle: {cycle}")
        if RI_update > early_stop and MI_update > early_stop:
            print(f"Stopped early on cycle: {cycle}")
            return data
        # update previous
        data['new_clust_prev'] = data['new_clust']

    return data


def gp_cluster_isotropic_rbf_lambda_earlyStop_RI_MI_nugget(data, s_list, x_list, y_list, num_clusters, num_cycles, constant_lb, constant_ub, length_lb, length_ub, lambda_=0, early_stop=0.9):
    import math
    from sklearn.gaussian_process.kernels import WhiteKernel
    # initialize bounded constant*RBF sklearn kernel
    kernel = ConstantKernel(constant_value_bounds=(constant_lb, constant_ub)) * RBF(length_scale_bounds=(length_lb, length_ub)) + WhiteKernel(1e-1, noise_level_bounds=(1e-5, 1e2))

    # initialize GPs
    gp_instance = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # combine s and x domain for isotropic kernel
    gp_fit_list = np.append(s_list, x_list)
    new_data = data[gp_fit_list]
    data['new_clust_prev'] = data['new_clust']

    for cycle in range(1, num_cycles + 1):
        for i in range(1, num_clusters + 1):
            x_train_i = data.loc[data['new_clust'] == i][gp_fit_list] # loc to find =2, ['x1'])
            y_train_i = data.loc[data['new_clust'] == i][y_list]

            if len(x_train_i) == 0:
                raise ValueError('Empty Cluster Detected')

            # fit each GP
            gp_fit = gp_instance.fit(x_train_i, y_train_i)
            print(f"GPSC Round: {cycle}, Clust_{i} Kernel = {gp_fit.kernel_}")
            data['gp'+ str(i) + '_mean_pred'] = gp_fit.predict(new_data, return_std=False)

        # calculate the centers
        center_i_s1 = []
        center_i_s2 = []
        for clust in range(1, num_clusters + 1):
            cluster_i_s1 = data.loc[data['new_clust'] == clust]['long'].to_numpy()  # loc to find =2, ['x1'])
            cluster_i_s2 = data.loc[data['new_clust'] == clust]['lat'].to_numpy()
            if len(cluster_i_s1) == 0:
                raise ValueError('Empty Cluster Detected')
            center_i_s1.append(sum(cluster_i_s1) / len(cluster_i_s1))
            center_i_s2.append(sum(cluster_i_s2) / len(cluster_i_s2))

        # update clusters
        for j in range(0, len(data.index)):
            diff_list = []
            for i in range(1, num_clusters + 1):
                diff_i = abs(data[y_list[0]][j] - data['gp' + str(i) + '_mean_pred'][j]) + (lambda_ * math.sqrt(abs( data['long'][j]- center_i_s1[i-1])**2 + abs( data['lat'][j]- center_i_s2[i-1])**2))
                diff_list.append(diff_i)

            # find the cluster of the min error
            data.iloc[j, data.columns.get_loc('new_clust')] = diff_list.index(min(diff_list)) + 1

        RI_update = metrics.adjusted_rand_score(data['new_clust_prev'], data['new_clust'])
        MI_update = metrics.adjusted_mutual_info_score(data['new_clust_prev'], data['new_clust'])
        print(f" RI_update = {round(RI_update, 3)}, MI_update = {round(MI_update, 3)}, for cycle: {cycle}")
        if RI_update > early_stop and MI_update > early_stop:
            print(f"Stopped early on cycle: {cycle}")
            return data
        # update previous
        data['new_clust_prev'] = data['new_clust']

    return data
