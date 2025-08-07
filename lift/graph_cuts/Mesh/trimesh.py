import numpy as np

from .utils.hedge import HEdge
from .utils import *


class TriMesh:
    """
    Triangle mesh object with access to the vertices and triangles.
    """

    f_dtype = np.dtype([
        ("triangles", 'i4', (3, )),
        ("normals", 'f8', (3, )),
        ("areas", 'f8', (1, )),
        ("labels", 'i4', (1, )),
    ])

    v_dtype = np.dtype([
        ("vertices", 'f8', (3, )),
        ("v_labels", 'i4', (1, )),
    ])

    def __init__(self, vertices:np.ndarray, triangles:np.ndarray, labels: np.ndarray=None,
                 remove_duplicated_vertices=True):
        self.v_data = np.empty(vertices.shape[0], dtype=self.v_dtype, order='F')
        self.v_data['vertices'] = vertices
        self.f_data = np.empty(triangles.shape[0], dtype=self.f_dtype, order='F')
        self.f_data['triangles'] = triangles

        if labels is None:
            self.f_data['labels'] = 0
        else:
            self.f_data['labels'][:, 0] = labels.astype('i4')

        if remove_duplicated_vertices:
            self.remove_duplicated_vertices()
            self.remove_unreferenced_vertices(allocate_memory=True)

        self.update_normals()
        self.update_areas()

    def dilate_labels(self, label=None, itr=1, mode="e"):
        """
        Dilate labeled triangles by one ring of adjacent triangles per iteration.
        """
        if label is None: label = list(range(1, self.n_class))
        label = [label] if np.array(label).shape==() else label
        labels = self.labels
        for _ in range(itr):
            for i in label:
                lt_idx = np.where(self.labels==i)[0]
                if len(lt_idx) == 0: continue
                t_idx = self.get_adjacent_traingles(lt_idx, mode=mode)
                labels[t_idx] = i

    def get_boundary_triangles(self, idx=None):
        if not hasattr(self, "hedges"):
            self.update_hedges()
        if idx is None:
            return self.hedges.get_boundary_triangles()
        else:
            assert len(idx)>0, "idx is empty!"
            idx = np.asarray(idx)
            he = HEdge(self.triangles[idx])
            return idx[he.get_boundary_triangles()]

    def get_adjacent_traingles(self, t_idx, mode='e'):
        """
        Get adjacent triangles, exclusive of t_idx. Adjacency is defined by edges.
        """
        t_idx = [t_idx] if np.array(t_idx).shape==() else t_idx
        assert len(t_idx) > 0, "t_idx is empty!"
        if not hasattr(self, "hedges"):
            self.update_hedges()
        
        if mode != 'e':
            raise NotImplementedError("Only edge adjacency ('e') is supported.")

        tt_adj_mat = self.hedges.get_tt_adjacent_matrix()
        adj_t_idx = set(tt_adj_mat[t_idx].reshape(-1))
        adj_t_idx = adj_t_idx - {-1} - set(t_idx)
        return list(adj_t_idx)

    def remove_triangles(self, idx_to_remove, allocate_memory=False):
        if len(idx_to_remove) == 0: return
        reserved = np.ones(self.n_f, dtype='bool')
        reserved[idx_to_remove] = False
        if allocate_memory:
            tmp = np.empty(reserved.sum(), dtype=self.f_dtype)
            tmp[:] = self.f_data[reserved]
            self.f_data = tmp
        else:
            self.f_data = self.f_data[reserved]

        if hasattr(self, "hedges"):
            self.hedges.remove_triangles(idx_to_remove)

    def remove_vertices(self, idx_to_remove, remove_triangles=True, allocate_memory=False):
        reserved = np.ones(self.n_v, dtype='bool')
        idx_mapping_vector = -np.ones(self.n_v, dtype='int')

        reserved[idx_to_remove] = False
        if allocate_memory:
            tmp = np.empty(reserved.sum(), dtype=self.v_dtype)
            tmp[:] = self.v_data[reserved]
            self.v_data = tmp
        else:
            self.v_data = self.v_data[reserved]
        
        idx_mapping_vector[reserved] = np.arange(reserved.sum())
        self.triangles = idx_mapping_vector[self.triangles]
        
        if remove_triangles:
            tri_idx_to_remove = np.where(np.any(self.triangles==-1, 1))[0]
            self.remove_triangles(tri_idx_to_remove)

    def remove_duplicated_vertices(self):
        idx_sorted = np.lexsort(self.vertices.T)
        is_unique = np.empty(self.n_v, dtype='bool')
        if self.n_v > 0:
            is_unique[0] = True
            is_unique[1:] = np.any(self.vertices[idx_sorted[1:]] != self.vertices[idx_sorted[:-1]], axis=1)
        if self.n_v == 0 or np.all(is_unique): return

        sorted_idx_mapping = np.arange(self.n_v) * is_unique
        unique_idx = sorted_idx_mapping[is_unique]
        last_i = 0
        for cur_i in unique_idx[1:]:
            sorted_idx_mapping[last_i:cur_i] = sorted_idx_mapping[last_i]
            last_i = cur_i
        sorted_idx_mapping[unique_idx[-1]:] = sorted_idx_mapping[unique_idx[-1]]

        vert_idx_mapping = np.arange(self.n_v)
        vert_idx_mapping[idx_sorted] = vert_idx_mapping[idx_sorted[sorted_idx_mapping]]
        self.triangles = vert_idx_mapping[self.triangles]

    def remove_unreferenced_vertices(self, allocate_memory=False):
        if self.n_f == 0:
            if self.n_v > 0:
                self.remove_vertices(np.arange(self.n_v), allocate_memory=allocate_memory)
            return
            
        being_removed = np.ones(self.n_v, dtype='bool')
        being_removed[self.triangles.reshape(-1)] = False
        vert_idx_to_remove = np.where(being_removed)[0]
        if len(vert_idx_to_remove) > 0:
            self.remove_vertices(vert_idx_to_remove, allocate_memory=allocate_memory)

    def update_hedges(self):
        self.hedges = HEdge(self.triangles)

    def update_normals(self):
        if self.n_f == 0: return
        self.normals[:] = np.cross(self.vertices[self.v1] - self.vertices[self.v0],
                                   self.vertices[self.v2] - self.vertices[self.v0])
        self.normals /= np.sum(self.normals**2, 1, keepdims=True)**0.5 + 1e-9

    def update_areas(self):
        if self.n_f == 0: 
            self.areas.fill(0)
            return
        normals = np.cross(self.vertices[self.v1] - self.vertices[self.v0],
                           self.vertices[self.v2] - self.vertices[self.v0])
        self.areas = 0.5 * np.sqrt((normals**2).sum(axis=1))

    def update_v_labels(self):
        self.v_labels.fill(0)
        for i in range(1, self.n_class):
            idx = self.triangles[np.where(self.labels==i)].reshape(-1)
            self.v_labels[idx] = i

    def update_t_labels(self):
        """
        Update triangle labels from vertex labels. A labeled triangle must contain three same labeled vertices.
        """
        if self.n_v > 0:
            self.labels = np.min(self.v_labels[self.triangles], 1)
        else:
            self.labels.fill(0)

    def clear(self):
        if hasattr(self, "hedges"): del self.hedges

    @property
    def vertices(self):
        return self.v_data['vertices']

    @vertices.setter
    def vertices(self, v):
        self.v_data['vertices'] = v

    @property
    def v_labels(self):
        return self.v_data['v_labels'][:, 0]

    @v_labels.setter
    def v_labels(self, v):
        self.v_data['v_labels'][:, 0] = v

    @property
    def labels(self):
        return self.f_data['labels'][:, 0]

    @labels.setter
    def labels(self, v):
        self.f_data['labels'][:, 0] = v

    @property
    def triangles(self):
        return self.f_data['triangles']

    @triangles.setter
    def triangles(self, v: np.ndarray):
        self.f_data['triangles'] = v
        self.clear()

    @property
    def normals(self):
        return self.f_data['normals']

    @normals.setter
    def normals(self, v=np.ndarray):
        self.f_data['normals'] = v

    @property
    def areas(self):
        return self.f_data['areas'][:, 0]

    @areas.setter
    def areas(self, v):
        self.f_data['areas'][:, 0] = v

    @property
    def v0(self):
        return self.triangles[:, 0]

    @property
    def v1(self):
        return self.triangles[:, 1]

    @property
    def v2(self):
        return self.triangles[:, 2]
        
    @property
    def centers(self):
        return self.vertices[self.triangles].mean(1)

    @property
    def n_v(self):
        return len(self.vertices)

    @property
    def n_f(self):
        return len(self.triangles)

    @property
    def n_class(self):
        if self.n_f == 0 or len(self.labels) == 0:
            return 1
        return np.max(self.labels) + 1
