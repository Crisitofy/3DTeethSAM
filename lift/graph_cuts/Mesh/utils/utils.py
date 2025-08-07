import numpy as np

_epsilon = 1e-7     # float64
_max_size = 1e8
MAX = 1e16


def get_dist_from_point_to_triangles(point, triangles):
    """
    Get distances from one point to triangles.
    :param point: (3, )
    :param triangles: (n, 3, 3)
    :return: dists: (n, 3)
    """
    projections, _ = project_to_triangle(point, triangles)
    projections = projections[0]
    inside = check_inside_triangles(projections, triangles)
    dists1 = np.sum((point[None, :] - projections)**2, 1)**0.5
    dists1[inside==False] = np.finfo('d').max
    # dists1[inside==False] = 10000.
    assert np.all(dists1 >= 0.)
    dists2 = np.sum((point[None, None :] - triangles)**2, 2).min(1)**0.5
    assert np.all(dists2 >= 0.)
    return np.minimum(dists1, dists2)


def calc_spherical_coords(xyz: np.ndarray):
    """
    Calculate the spherical coordinates according to Cartesian coordinates.
    Rho>=0, 0<=theta<=2*pi, 0<=phi<=pi
    :param xyz: Cartesian coordinates, (n, 3), (x, y, z)
    :return: pts: Spherical coordinates, (n, 3), (rho, theta, phi)
    """
    pts = np.empty(xyz.shape, dtype='float64')
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    pts[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    pts[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])
    pts[:, 2] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # elevation angle(theta)
    return pts


def project_to_plane(points: np.ndarray, ref_points:np.ndarray, ref_normals:np.ndarray, proj_vects:np.ndarray=None):
    """
    Project points to each plane determined by reference points and reference normals,
    along the direction of projection vector.
    If the plane is in the reverse direction of projection vector, k is < 0.
    :param points: (n, 3) or (3, )
    :param ref_points: (m, 3)
    :param ref_normals: (m, 3)
    :param proj_vects: (n, 3), or (3, )
    :return: (ret_points, k): ((n, m, 3), (n, m))
    """
    if points.shape == (3, ): points = points[None, :]

    n, m = len(points), len(ref_points)
    assert n * m < _max_size, "number of points and planes exceeds limit!"

    points = points[:, None, :]  # (n, 1, 3)
    ref_points = ref_points[None, :, :]  # (1, m, 3)
    ref_normals = ref_normals[None, :, :]  # (1, m, 3)

    if proj_vects is None:
        proj_vects = ref_normals        # (1, m, 3)
    else:
        if proj_vects.shape == (3, ): proj_vects = proj_vects[None, :]
        proj_vects = proj_vects[:, None, :]     # (n, 1, 3)

    k = np.sum((ref_points - points)*ref_normals, -1) / (np.sum(ref_normals*proj_vects, -1) + _epsilon) # (n, m)
    return points + k[:, :, None] * proj_vects, k


def project_to_triangle(points, triangles, proj_vects=None):
    """
    Project points to each planes determined by triangles, along the direction of
    projection vectors, which are the normals of triangles by default.
    :param points: (n, 3), or (3, )
    :param triangles: (m, 3, 3)
    :param proj_vects: (n, 3), or (3, ), None by default
    :return: (ret_points, k): ((n, m, 3), (n, m)), where ret_p = p + k * proj_vect
    """
    normals = _calc_normals(triangles)
    return project_to_plane(points, triangles[:, 0], normals, proj_vects)


def check_inside_triangles(points:np.ndarray, triangles:np.ndarray):
    """
    Check if points are inside corresponding triangles.
    :param points: (n, 3)
    :param triangles: (n, 3, 3)
    :return: inside: bool, (n, )
    """
    assert triangles.shape[1:] == (3, 3)
    v = points[:, None, :] - triangles
    v1 = np.cross(v[:, 0, :], v[:, 1, :])
    v2 = np.cross(v[:, 1, :], v[:, 2, :])
    v3 = np.cross(v[:, 2, :], v[:, 0, :])
    # is point in the plane where the triangle is
    is_in_plane = np.all(abs(np.cross(v1, v2)) < _epsilon, axis=1)
    is_direct_same12 = np.sum(v1*v2, axis=1)>=0
    is_direct_same13 = np.sum(v1*v3, axis=1)>=0
    return is_in_plane * is_direct_same12 * is_direct_same13


def _calc_normals(triangles):
    return np.cross(triangles[:, 1]-triangles[:, 0],
                    triangles[:, 2]-triangles[:, 0])

