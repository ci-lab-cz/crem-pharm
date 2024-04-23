import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

from pmapper.pharmacophore import Pharmacophore as P


class PharmModel2(P):

    def __init__(self, bin_step=1, cached=False):
        super().__init__(bin_step, cached)
        self.clusters = defaultdict(set)
        self.exclvol = None

    def get_num_features(self):
        return len(self._get_ids())

    def get_xyz(self, ids):
        return np.array([xyz for label, xyz in self.get_feature_coords(ids)])

    def set_clusters(self, clustering_threshold, init_ids=None):
        """

        :param clustering_threshold:
        :param init_ids: ids of features selected as a starting pharmacophore
        :return:
        """
        ids = tuple(set(self._get_ids()) - set(init_ids))
        coords = self.get_xyz(ids)
        c = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_threshold).fit(coords)
        for i, j in enumerate(c.labels_):
            self.clusters[j].add(ids[i])

    def get_subpharmacophore(self, ids):
        coords = self.get_feature_coords(ids)
        p = P()
        p.load_from_feature_coords(coords)
        return p

    def select_nearest_cluster(self, ids):

        def min_distance(ids1, ids2):
            xyz1 = self.get_xyz(ids1)
            xyz2 = self.get_xyz(ids2)
            return np.min(cdist(xyz1, xyz2))

        ids = set(ids)
        selected_ids = tuple()
        min_dist = float('inf')
        for k, v in self.clusters.items():
            if not v & set(ids):
                dist = min_distance(ids, v)
                if dist < min_dist:
                    min_dist = dist
                    selected_ids = tuple(v)
        return selected_ids

    def get_feature_coords_pd(self, ids=None):
        ids = self._get_ids(ids)
        data = [(i, label, x, y, z) for i, (label, (x, y, z)) in zip(ids, self.get_feature_coords(ids))]
        coords = pd.DataFrame(data, columns=['id', 'label', 'x', 'y', 'z'])
        return coords

    def load_from_xyz(self, fname):
        self.exclvol = []
        with open(fname) as f:
            feature_coords = []
            f.readline()
            line = f.readline().strip()
            if line:
                opts = dict(item.split('=') for item in line.split(';'))
                if 'bin_step' in opts:
                    self.update(bin_step=float(opts['bin_step']))
            for line in f:
                label, *coords = line.strip().split()
                coords = tuple(map(float, coords))
                if label != 'e':
                    feature_coords.append((label, coords))
                else:
                    self.exclvol.append(coords)
            self.load_from_feature_coords(tuple(feature_coords))
        if self.exclvol:
            self.exclvol = np.array(self.exclvol)
        else:
            self.exclvol = None


