"""
Face-to-face adjacency on any mesh, found from cell geometry rather than
shared node ids.
"""

import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree

from EarthSystemGrids.base.UnstructuredGridMesh import _FILL_VALUE


def lonlat_to_xyz(lon_deg, lat_deg):
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def face_adjacency(mesh, tol=1e-9):
    """Symmetric sparse (nface, nface) graph: two faces are neighbours when
    they share an edge. Shared edges are found by coordinates (edge
    midpoints on the unit sphere coincide), not by node ids, so faces on
    either side of a longitude seam -- whose nodes this repo's meshes do not
    merge -- are still connected. Degenerate edges (both corners on the same
    point, e.g. a pole) and unused `face_nodes` slots connect nothing."""
    face_nodes = mesh.face_nodes  # (nface, nc)
    nface, nc = face_nodes.shape
    corners = lonlat_to_xyz(np.rad2deg(mesh.node_lon[face_nodes]), np.rad2deg(mesh.node_lat[face_nodes]))
    a = corners
    b = np.roll(corners, -1, axis=1)
    used = face_nodes != _FILL_VALUE
    midpoints = (0.5 * (a + b)).reshape(-1, 3)
    valid = ((np.linalg.norm(a - b, axis=-1) > tol) & used & np.roll(used, -1, axis=1)).reshape(-1)
    edge_face = np.repeat(np.arange(nface), nc)

    idx = np.flatnonzero(valid)
    pairs = cKDTree(midpoints[idx]).query_pairs(r=tol, output_type="ndarray")
    fi = edge_face[idx[pairs[:, 0]]]
    fj = edge_face[idx[pairs[:, 1]]]
    keep = fi != fj
    fi, fj = fi[keep], fj[keep]
    rows = np.concatenate([fi, fj])
    cols = np.concatenate([fj, fi])
    adj = scipy.sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(nface, nface))
    adj.data[:] = 1.0  # collapse duplicate pairs
    return adj
