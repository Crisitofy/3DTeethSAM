import numpy as np
import maxflow
from lift.graph_cuts.Mesh.trimesh import TriMesh

def get_angular_distance(m:TriMesh, edges):
    i, j = edges[:, 0], edges[:, 1]
    angular_dists = 1. - np.sum(m.normals[i] * m.normals[j], 1)
    angular_dists = np.clip(angular_dists, 0., 2.)
    return angular_dists

def reflex_angles(m:TriMesh, edges):
    # return a boolean array
    c12 = m.centers[edges[:, 1]] - m.centers[edges[:, 0]]
    n1 = m.normals[edges[:, 0]]
    return np.sum(c12*n1, 1)>0

class FuzzyClustering:
    """
    :param yita: angular distance ::= yita * (1- cos(a)) if the interior angle < 180;
                                      1-cos(a), else.
    """

    def __init__(self, yita=0.05):
        self.yita = yita

    def solve(self, m:TriMesh, tooth_nodes, gum_nodes):
        """
        :param m:
        :param tooth_nodes: absolutely belong tooth.
        :param gum_nodes: absolutely belong to gum.
        :param fuzzy_nodes: including the tooth-gum boundary.
        :return:
        """
        g = maxflow.Graph[int]()
        g.add_nodes(m.n_f)
        edges = self.get_edges(m)
        caps = self.get_capacities(m, edges) * 1000

        g.add_edges(edges[:, 0], edges[:, 1], caps, caps)
        for i in list(tooth_nodes):
            g.add_tedge(i, 0, 1000000)  # tooth as sink
        for i in list(gum_nodes):
            g.add_tedge(i, 1000000, 0)  # gum as source

        g.maxflow()

        ret = []
        for i in range(m.n_f):
            if g.get_segment(i):
                ret.append(i)
        ret = np.array(ret, dtype='int64')
        return ret

    def get_edges(self, m:TriMesh):
        m.update_hedges()
        tt = m.hedges.get_tt_adjacent_matrix()
        edges = np.zeros((3 * m.n_f, 2), dtype=int)
        edges[:, 0] = np.arange(3 * m.n_f) // 3
        edges[:, 1] = tt.reshape(-1)
        edges = edges[np.where(edges[:, 1] >= 0)[0], :]
        return edges[edges[:, 0]<edges[:, 1]]

    def get_capacities(self, m:TriMesh, edges):
        yita = np.ones(len(edges))
        yita[np.logical_not(reflex_angles(m, edges))] = self.yita

        angular_dists = get_angular_distance(m, edges) * yita
        avg_ad = angular_dists.mean()
        cap = 1. / (1.+angular_dists/avg_ad)

        areas = np.sum(m.areas[edges], axis=1)
        cap += areas * 50

        return cap


class Teeth(TriMesh):

    def fuzzy_cluster(self, label):
        ori_labels = self.labels.copy()
        boundary_idx = self.get_boundary_triangles(np.where(self.labels==label)[0])

        self.labels = 0
        self.labels[boundary_idx] = 1
        self.dilate_labels(1, 5)

        is_selected = self.labels==1
        select_idx = np.where(self.labels == 1)[0]
        select_m = TriMesh(self.vertices, self.triangles[select_idx], ori_labels[select_idx]==label)
        self.segment(select_m)

        self.labels = ori_labels
        self.labels[np.logical_and(self.labels==label, is_selected)] = 0
        self.labels[select_idx[select_m.labels>0]] = label

    def segment(self, select_m):
        fc = FuzzyClustering(0.05)
        idx = select_m.get_boundary_triangles()
        is_boundary = np.zeros(select_m.n_f)
        is_boundary[idx] = 1

        tooth_nodes = np.where(np.logical_and(is_boundary, select_m.labels>0))[0]
        gum_nodes = np.where(np.logical_and(is_boundary, select_m.labels==0))[0]
        ret_idx = fc.solve(select_m, tooth_nodes, gum_nodes)
        select_m.labels = 0
        select_m.labels[ret_idx] = 1
