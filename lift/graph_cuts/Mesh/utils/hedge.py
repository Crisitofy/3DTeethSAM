# -*- coding: utf-8 -*-
"""
hedge.py

@author: kaidi
Half edge class.
"""
import numpy as np


class HEdge:

    dtype = np.dtype([
        ("start", "i4", (1, )),         # the index of start vertex
        ("prev", "i4", (1, )),          # the index of previous half edge in a triangle
        ("next", "i4", (1, )),          # the index of next half edge in a triangle
        ("twin", "i4", (1, )),          # the index of twin half edge
        ("belong", "i4", (1, )),        # the index of belonging triangle
        ("valid", "bool", (1, ))        # whether the edge is valid
        # ("visited", "bool", (1, )),
        # ("non_manifold", "bool", (1, )) # if the edge is non-manifold
    ])

    def __init__(self, triangles:np.ndarray):
        self._construct_half_edges(triangles)

    def _construct_half_edges(self, triangles:np.ndarray):
        """
        Construct half edges from vertices and triangles.
        :param triangles: np.array, (m, 3), int. Made up of indices of vertices.
        :return: half edges: np.array with self.dtype
        """
        self.triangles = triangles
        self.hedges = np.empty(len(triangles)*3, dtype=self.dtype, order='F')
        self.valid = True
        idx = np.arange(self.n)
        self.start = triangles.reshape(-1)
        self.belong = idx // 3
        self.prev = idx.reshape((-1, 3))[:, [2, 0, 1]].reshape(-1)
        self.next = idx.reshape((-1, 3))[:, [1, 2, 0]].reshape(-1)
        self.update_twin()

    def update_twin(self, edges=None):
        """
        Update all valid edges' twins.
        :param edges: must be valid edges
        """
        edges = np.where(self.valid)[0] if edges is None else edges
        if len(edges) == 0: return
        if not hasattr(self, "vert_edge_map"):
            self.update_vert_edge_map()

        assert np.all(self.valid[edges]), "Exist invalid edges."
        self.twin[edges] = -1      # no twin, representing boundaries
        twin = self.twin
        start = self.start
        end = self.end
        visited = np.ones(self.n, dtype='bool')
        visited[edges] = False
        for e1 in edges:
            if visited[e1]: continue
            for e2 in self.vert_edge_map[end[e1]]:
                if end[e2] == start[e1]:
                    twin[e1] = e2
                    twin[e2] = e1
                    visited[e2] = True
            visited[e1] = True

    def update_vert_tri_map(self):
        vert_tri_map = [[] for _ in range(np.max(self.start)+1)]
        for v, t in zip(self.start[self.valid], self.belong[self.valid]):
            vert_tri_map[v].append(t)
        self.vert_tri_map = vert_tri_map

    def update_vert_edge_map(self):
        vert_edge_map = [[] for _ in range(np.max(self.start)+1)]
        valid = self.valid
        for e, v in enumerate(self.start):
            if valid[e]: vert_edge_map[v].append(e)
        self.vert_edge_map = vert_edge_map

    def update_tri_edge_matrix(self):
        tri_edge_matrix = np.lexsort((self.belong, np.logical_not(self.valid))).reshape((-1, 3))[:len(self.triangles)]
        # remove invalid edges
        self.tri_edge_matrix = tri_edge_matrix

    def remove_triangles(self, tri_idx_to_remove):
        # update twin
        # set edges to invalid
        if len(tri_idx_to_remove)==0: return
        edges_to_remove = self.get_edges_from_triangles(tri_idx_to_remove).reshape(-1)
        assert np.all(self.valid[edges_to_remove]), "Exist invalid edges."
        self.valid[edges_to_remove] = False
        self.clear()

        # update belong
        tri_map = np.arange(len(self.triangles))
        tri_map[tri_idx_to_remove] = -1
        remained = np.where(tri_map>=0)[0]
        tri_map[remained] = np.arange(len(remained))
        self.belong = tri_map[self.belong]

        # update triangles
        self.triangles = self.triangles[remained]

        edges_with_twin = np.where(np.logical_and(self.valid, self.twin>=0))[0]
        twin_invalid = np.logical_not(self.valid[self.twin[edges_with_twin]])
        edges_to_update = edges_with_twin[twin_invalid]
        self.update_twin(edges_to_update)

    def get_connected_components(self, tri_idx=None, mode='e'):
        """
        :param tri_idx: list
        :param mode: 'e'(edge) or 'v'(vertex)
        :return clique_list: a sorted list of cliques, each containing a list of triangle indices.
        """
        tri_idx = [i for i in range(len(self.triangles))] if tri_idx is None else tri_idx
        if len(tri_idx) == 0: return []
        assert len(tri_idx) > 0, "tri_idx is empty!"
        tri_indices = -np.ones(len(self.triangles), dtype='int')
        tri_indices[tri_idx] = tri_idx

        if mode == 'e':
            tt_adj_mat = self.get_tt_adjacent_matrix()
        else:
            vt_adj_mat = self.get_vt_adjacent_matrix()
            tt_adj_mat = vt_adj_mat[self.triangles].reshape((len(self.triangles), -1))
        idx_minus1 = np.where(tt_adj_mat<0)
        # tt_adj_mat = tri_indices[tt_adj_mat[tri_idx]] # only include tri_idx
        tt_adj_mat = tri_indices[tt_adj_mat] # only include tri_idx
        tt_adj_mat[idx_minus1] = -1

        visited = np.ones(len(self.triangles), dtype='bool')
        visited[tri_idx] = False
        clique_list = []
        while np.any(visited == False):
            clique = []
            stack = []
            init = np.where(visited == False)[0][0]
            stack.append(init)
            clique.append(init)
            visited[init] = True
            while len(stack) > 0:
                # t = stack.pop()
                # for adj_t in tt_adj_mat[t]:
                #     if adj_t >= 0 and not visited[adj_t]:
                #         stack.append(adj_t)
                #         clique.append(adj_t)
                #         visited[adj_t] = True
                adj_t_list = np.asarray(list(set(tt_adj_mat[stack].reshape(-1)) - {-1}))
                stack.clear()
                if len(adj_t_list) == 0: break
                adj_t_list = adj_t_list[np.logical_not(visited[adj_t_list])]
                stack = list(adj_t_list)
                clique.extend(adj_t_list)
                visited[adj_t_list] = True

            clique_list.append(clique)

        # sort: descending
        idx = np.argsort([-len(c) for c in clique_list])
        return [clique_list[i] for i in idx]

    def get_non_manifold_edges(self):
        """Return a list of the indices of non-manifold edges"""
        if hasattr(self, "non_manifold_edges"):
            return self.non_manifold_edges

        if not hasattr(self, "vert_edge_map"):
            self.update_vert_edge_map()

        edges = np.where(self.valid)[0]
        start = self.start[edges]
        end = self.end[edges]
        idx_sorted = np.lexsort((end, start))
        identical = np.logical_and(start[idx_sorted[1:]] == start[idx_sorted[:-1]],
                                   end[idx_sorted[1:]] == end[idx_sorted[:-1]]
                                   )
        identical_a = np.concatenate(([False], identical))
        identical_b = np.concatenate((identical, [False]))

        self.non_manifold_edges = edges[idx_sorted[identical_a | identical_b]]
        return self.non_manifold_edges

    def get_non_manifold_triangles(self):
        non_manifold_edges = self.get_non_manifold_edges()
        assert np.all(self.valid[non_manifold_edges]), "Exist invalid edges."
        return np.asarray(list(set(self.belong[non_manifold_edges])))

    def get_non_manifold_vertices(self):
        non_manifold_edges = self.get_non_manifold_edges()
        assert np.all(self.valid[non_manifold_edges]), "Exist invalid edges."
        return list(set(self.start[non_manifold_edges]).union(set(self.end[non_manifold_edges])))

    def has_boundaries(self):
        return np.any(self.twin[self.valid]<0)

    def get_boundaries_edges(self):
        return np.where(np.logical_and(self.valid, self.twin==-1))[0]

    def get_boundary_vertices(self):
        bound_edges = self.get_boundaries_edges()
        assert np.all(self.valid[bound_edges]), "Exist invalid edges."
        return list(set(self.start[bound_edges]))

    def get_boundary_triangles(self):
        bound_edges = self.get_boundaries_edges()
        assert np.all(self.valid[bound_edges]), "Exist invalid edges."
        return list(set(self.belong[bound_edges]))

    def get_vv_adjacent_matrix(self):
        """Get vertex-vertices adjcent matrix.
        This may include duplicate adjacent vertices(non-manifold).
        The 1st column is the number of adjacent vertices. -1 indicates invalid.
        """
        if hasattr(self, "_vv_adj_mat"):
            return self._vv_adj_mat

        vv_adj_list = [[] for _ in range(np.max(self.start) + 1)]
        for v, adj_v in zip(self.start[self.valid], self.end[self.valid]):
            vv_adj_list[v].append(adj_v)

        num_adj = [len(l) for l in vv_adj_list]
        vv_adj_mat = -np.ones((len(num_adj), max(num_adj)+1), dtype='int')
        vv_adj_mat[:, 0] = num_adj
        for i, adj_idx in enumerate(vv_adj_list):
            vv_adj_mat[i, 1:num_adj[i]+1] = adj_idx

        self._vv_adj_mat = vv_adj_mat
        return self._vv_adj_mat

    def get_vt_adjacent_matrix(self):
        """
        Get vertex-triangles adjacent matrix.
        The 1st column is the number of adjacent triangles. -1 indicates invalid.
        """
        if hasattr(self, "_vt_adj_mat"): return self._vt_adj_mat
        if not hasattr(self, "vert_tri_map"): self.update_vert_tri_map()
        num_adj = [len(l) for l in self.vert_tri_map]
        vt_adj_mat = -np.ones((len(num_adj), max(num_adj)+1), dtype='int')
        vt_adj_mat[:, 0] = num_adj
        for i, adj_idx in enumerate(self.vert_tri_map):
            vt_adj_mat[i, 1:num_adj[i]+1] = adj_idx

        self._vt_adj_mat = vt_adj_mat
        return self._vt_adj_mat

    def get_triangles_containing_vertices(self, v_idx):
        """
        Get triangles containing one or more of v_idx
        :param v_idx: indices of vertices
        :return:
        """
        if not hasattr(self, "vert_tri_map"):
            self.update_vert_tri_map()
        if isinstance(v_idx, int):
            v_idx = [v_idx]

        assert len(v_idx)> 0, "v_idx is empty!"
        ret_idx =[]
        for v_i in v_idx:
            ret_idx.extend(self.vert_tri_map[v_i])
        return list(set(ret_idx))

    def get_edges_from_triangles(self, t_idx):
        assert isinstance(t_idx, int) or len(t_idx) > 0, "t_idx is empty!"
        if not hasattr(self, "tri_edge_matrix"):
            self.update_tri_edge_matrix()

        edges = self.tri_edge_matrix[t_idx]
        if len(edges)>0:
            assert np.all(self.valid[edges]), "Exist invalid edges"
        return edges

    def get_tt_adjacent_matrix(self):
        """
        Get triangle-triangle adjacent(sharing same edge) matrix.
        One triangle has only three adjacent triangles.
        :return tt_adjacent_matrix: np.array (n, 3), -1 indicating no adjacent triangle.
        """
        if hasattr(self, "_tt_adjacent_matrix"):
            return self._tt_adjacent_matrix

        if not hasattr(self, "tri_edge_matrix"):
            self.update_tri_edge_matrix()

        assert np.all(self.valid[self.tri_edge_matrix]), "Exist invalid edges."

        tri_twins = self.twin[self.tri_edge_matrix]
        idx = np.where(tri_twins<0) # no twin

        tri_twins[idx] = 0
        tt_adjacent_matrix = self.belong[tri_twins]
        tt_adjacent_matrix[idx] = -1

        self._tt_adjacent_matrix = tt_adjacent_matrix
        return self._tt_adjacent_matrix

    def clear(self):
        if hasattr(self, "vert_tri_map"): del self.vert_tri_map
        if hasattr(self, "vert_edge_map"): del self.vert_edge_map
        if hasattr(self, "tri_edge_matrix"): del self.tri_edge_matrix
        if hasattr(self, "non_manifold_edges"): del self.non_manifold_edges
        if hasattr(self, "_vv_adj_mat"): del self._vv_adj_mat
        if hasattr(self, "_tt_adjacent_matrix"): del self._tt_adjacent_matrix
        if hasattr(self, "_vt_adj_mat"): del self._vt_adj_mat

    @property
    def start(self):
        return self.hedges['start'][:, 0]

    @start.setter
    def start(self, v):
        self.hedges['start'][:, 0] = v

    @property
    def next(self):
        return self.hedges['next'][:, 0]

    @next.setter
    def next(self, v):
        self.hedges['next'][:, 0] = v

    @property
    def prev(self):
        return self.hedges['prev'][:, 0]

    @prev.setter
    def prev(self, v):
        self.hedges['prev'][:, 0] = v

    @property
    def twin(self):
        return self.hedges['twin'][:, 0]

    @twin.setter
    def twin(self, v):
        self.hedges['twin'][:, 0] = v

    @property
    def belong(self):
        return self.hedges['belong'][:, 0]

    @belong.setter
    def belong(self, v):
        self.hedges['belong'][:, 0] = v

    @property
    def valid(self):
        return self.hedges['valid'][:, 0]

    @valid.setter
    def valid(self, v):
        if np.any(self.hedges['valid'][:, 0] != v):
            self.hedges['valid'][:, 0] = v
            self.clear()

    @property
    def end(self):
        return self.start[self.next]

    @property
    def n(self):
        return len(self.hedges)


def _visualize(m, idx):

    COLORS = [[],               # reserved
              [1.0, 0.7, 0],    # yellow
              [0.952, 0.705, 0.627], # pink
              [0.615, 0.925, 0.513], # green
              [0.513, 0.764, 0.925], # blue
              [0.941, 0.678, 0.917], # purple
              ]

    vertices = np.asarray(m.vertices)
    triangles = np.asarray(m.triangles)
    labels = np.zeros(len(triangles), dtype='i4')
    labels[idx] = 1

    meshes = []
    for i in range(np.max(labels)+1):
        _idx = np.where(labels==i)[0]
        if len(_idx)==0:
            continue
        m = o3d.TriangleMesh()
        m.vertices = o3d.Vector3dVector(vertices)
        m.triangles = o3d.Vector3iVector(triangles[_idx])
        m.remove_unreferenced_vertices()
        m.compute_vertex_normals()
        if COLORS[i] != []:
            m.paint_uniform_color(COLORS[i])
        meshes.append(m)
    o3d.draw_geometries(meshes)


if __name__ == "__main__":
    import time
    import open3d as o3d
    m = o3d.read_triangle_mesh(R"stl")
    m.remove_duplicated_vertices()
    t = np.asarray(m.triangles)[:]
    he = HEdge(t)
    t1 = time.time()
    # he.update_vert_edge_map()
    mat = he.get_tt_adjacent_matrix()
    _visualize(m, mat[mat[10]].reshape(-1))
    # print(len(he.get_non_manifold_edges()))
    # print(len(he.get_non_manifold_edges()))
    # print(len(he.get_boundary_triangles()))

    # he.update_vert_edge_map()
    print(time.time()-t1)
