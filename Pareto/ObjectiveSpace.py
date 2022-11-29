import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pygmo import hypervolume
from pymoo.indicators.hv import HV
import os

np.random.seed(7)


class ObjectivesSpace:
    def __init__(self, df, functions):
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        # pts = obj_pts.copy()
        # obj_pts = obj_pts[obj_pts.sum(1).argsort()[::-1]]
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return pd.DataFrame(self.points[0], columns=self._constr_obj())

    def get_dominated(self):
        return pd.DataFrame(self.points[1], columns=self._constr_obj())

#    def _compute_hypervolume(self, x, r):
#        x = x[list(self.functions.keys())]
#        value = hypervolume([np.array(x)]).compute(r)
#        return value

    def _compute_hypervolume(self, x, r):
        x = x[list(self.functions.keys())]
        ind = HV(ref_point=r)
        # value = hypervolume([np.array(x)]).compute(r)
        return ind(np.array(x))

    def hypervolumes(self, r):
        factors = np.array(list(map(lambda x: -1 if x == 'max' else 1, list(self.functions.values()))))
        hv_pts = np.copy(self.points[0])
        hv_pts[:, 1:] = hv_pts[:, 1:] * factors
        # self.plot_provv(pts[:, 1:])
        not_dominated = pd.DataFrame(hv_pts, columns=self._constr_obj())
        not_dominated['hypervolume'] = \
            not_dominated.apply(lambda x: self._compute_hypervolume(x, r), axis=1)
        return not_dominated

    def knee_point(self):
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        kp_pts = np.copy(self.points[0])
        kp_pts[:, 1:] = kp_pts[:, 1:] * factors
        counter = np.zeros(kp_pts[:, 1:].shape[0])
        for i in range(100):
            w = np.random.uniform(low=0, high=1, size=kp_pts[:, 1:].shape)
            w = w / w.sum()
            counter[np.argmax((kp_pts[:, 1:] * w).sum(axis=1))] = counter[np.argmax((kp_pts[:, 1:] * w).sum(axis=1))] + 1
        columns = self._constr_obj() # .append('utility_based')
        columns.append('utility_based')
        # print(counter)
        return pd.DataFrame(np.column_stack((kp_pts, counter)), columns=columns)

    def euclidean_distance(self, up):
        ed_points = np.copy(self.points[0])
        ed_points[:, 1:] = (ed_points[:, 1:] - up) ** 2
        distances = ed_points[:, 1:].sum(axis=1) ** (1 / 2)
        # np.append(not_dominated, distances, axis=1)
        columns = self._constr_obj()  # .append('utility_based')
        columns.append('euclidean_distance')
        return pd.DataFrame(np.column_stack((ed_points, distances)), columns=columns)

    def weighted_mean(self, w):
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        wm_points = np.copy(self.points[0])
        wm_points[:, 1:] = wm_points[:, 1:] * factors
        wm_points[:, 1:] = wm_points[:, 1:] * w
        sum = wm_points[:, 1:].sum(axis=1) / wm_points[:, 1:].shape[1]
        columns = self._constr_obj()  # .append('utility_based')
        columns.append('weighted_mean')
        return pd.DataFrame(np.column_stack((wm_points, sum)), columns=columns)

    def _compute_distances(self, model, up, scale):
        relative_path = 'data/population'
        dir = os.listdir(relative_path)
        for el in dir:
            if model in el:
                model_per_user = el
                break
            else:
                model_per_user = ''
        df = pd.read_csv(relative_path + '/' + model_per_user, sep='\t')
        M = df[list(self.functions.keys())].values
        if scale:
            max = M.max(axis=0)
            min = M.min(axis=0)
            M = (M - min) / (max - min)
        M = (up - M) ** 2
        M = M.sum(axis=1) ** (1 / 2)
        # print(M.sum())
        return M.sum()

    def _variance(self, distances):
        mean = distances.mean()
        variance = ((distances - mean) ** 2).sum() / distances.shape[0]
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def _compute_distances_v2(self, model, up):
        relative_path = 'data/population'
        dir = os.listdir(relative_path)
        for el in dir:
            if model in el:
                model_per_user = el
                break
            else:
                model_per_user = ''
        df = pd.read_csv(relative_path + '/' + model_per_user, sep='\t')
        M = df[list(self.functions.keys())].values
        M = (up - M) ** 2
        M = M.sum(axis=1) ** (1 / 2)
        standard_deviation, variance = self._variance(M)
        # print(M.sum())
        # print(variance)
        print(variance, np.log(1 / (((2 * np.pi) ** (1 / 2)) * standard_deviation)), (1 / (2 * variance)) * M.sum())
        return M.shape[0] * np.log(1 / (((2 * np.pi) ** (1 / 2)) * standard_deviation)) - (1 / (2 * variance)) * M.sum()

    def nome_framework(self, up, scale=False):
        not_dominated = pd.DataFrame(self.points[0], columns=self._constr_obj())
        not_dominated['nostro_framework'] = not_dominated['model'].map(lambda x: self._compute_distances(x, up, scale))
        return not_dominated

    def nome_framework_v2(self, up):
        not_dominated = pd.DataFrame(self.points[0], columns=self._constr_obj())
        not_dominated['nostro_framework_v2'] = not_dominated['model'].map(lambda x: self._compute_distances_v2(x, up))
        return not_dominated

    def plot(self, not_dominated, dominated, r):
        not_dominated = not_dominated.values
        dominated = dominated.values
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(dominated[:, 1], dominated[:, 2], color='red')
        ax.scatter(not_dominated[:, 1], not_dominated[:, 2], color='blue')
        ax.scatter(r[0], r[1], color='green')
        # ax.plot(*not_dominated[not_dominated[:, 2].argsort()].T)
        #for i, txt in enumerate(not_dominated[:, 3]):
        #    ax.annotate(str(txt), (not_dominated[:, 1][i] * 1.005, not_dominated[:, 2][i]), fontsize=5)
        #for i, txt in enumerate(dominated[:, 3]):
        #    ax.annotate(str(txt), (dominated[:, 1][i] * 1.005, dominated[:, 2][i]), fontsize=5)
        # plt.axis([0, 1, 0, 1])
        plt.show()

    def plot_provv(self, not_dominated): #), dominated):
        # not_dominated = not_dominated.values
        # dominated = dominated.values
        fig = plt.figure()
        ax = fig.add_subplot()
        # ax.scatter(dominated[:, 1], dominated[:, 2], color='red')
        ax.scatter(not_dominated[:, 0], not_dominated[:, 1], color='blue')
        # ax.plot(*not_dominated[not_dominated[:, 2].argsort()].T)
        #for i, txt in enumerate(not_dominated[:, 3]):
        #    ax.annotate(str(txt), (not_dominated[:, 1][i] * 1.005, not_dominated[:, 2][i]), fontsize=5)
        #for i, txt in enumerate(dominated[:, 3]):
        #    ax.annotate(str(txt), (dominated[:, 1][i] * 1.005, dominated[:, 2][i]), fontsize=5)
        # plt.axis([0, 1, 0, 1])
        plt.show()

