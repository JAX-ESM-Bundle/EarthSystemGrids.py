"""
Surface-grid cleanup rules: land-sea mask from land fraction, required
straits, isolated-ocean filling, and topography/bathymetry consistent with
the mask (used by workflows/topobathymetry/04_ensure_featrues_of_topobathymetry_in_surface_grid.py).

All functions take flat arrays in the mesh's face order plus a mesh (e.g.
from `EarthSystemGrids.topobathymetry.io.read_mesh`) and an adjacency graph
from `EarthSystemGrids.base.adjacency.face_adjacency`.
Masks use the README convention: 1 = land, 0 = ocean.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from EarthSystemGrids.base.adjacency import face_adjacency, lonlat_to_xyz


@dataclass
class Strait:
    """A connection that must exist between two ocean basins, checked (and
    carved if missing) only through cells inside the lon/lat box -- so the
    carved channel cannot detour through some other continent."""

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    seed_a: tuple  # (lon, lat) of a point in the first basin
    seed_b: tuple  # (lon, lat) of a point in the second basin

    @classmethod
    def parse(cls, text):
        """NAME:LON_MIN,LON_MAX,LAT_MIN,LAT_MAX:LON_A,LAT_A:LON_B,LAT_B"""
        name, box, seed_a, seed_b = text.split(":")
        lon_min, lon_max, lat_min, lat_max = (float(v) for v in box.split(","))
        return cls(
            name, lon_min, lon_max, lat_min, lat_max,
            tuple(float(v) for v in seed_a.split(",")),
            tuple(float(v) for v in seed_b.split(",")),
        )


# Indonesian throughflow: Indian Ocean west of Sumatra <-> western Pacific
# north of New Guinea, through the Maritime Continent only.
INDONESIAN_THROUGHFLOW = Strait("indonesian_throughflow", 85.0, 155.0, -15.0, 15.0, (90.0, -5.0), (150.0, 5.0))

# On coarse grids these keep the Mediterranean, Red Sea, Black Sea and
# Hudson Bay connected to the open ocean; without them those seas are
# filled as land.
GIBRALTAR = Strait("gibraltar", 340.0, 10.0, 30.0, 42.0, (345.0, 36.0), (5.0, 38.0))
BAB_EL_MANDEB = Strait("bab_el_mandeb", 35.0, 55.0, 5.0, 25.0, (38.0, 20.0), (52.0, 12.0))
BOSPORUS = Strait("bosporus", 20.0, 42.0, 35.0, 47.0, (35.0, 43.0), (25.0, 37.0))
HUDSON_STRAIT = Strait("hudson_strait", 270.0, 305.0, 50.0, 68.0, (275.0, 58.0), (302.0, 58.0))


def _face_lonlat_deg(mesh):
    return np.rad2deg(mesh.face_lon), np.rad2deg(mesh.face_lat)


def _in_box(mesh, strait):
    """Box longitudes are taken mod 360; lon_min > lon_max means the box
    crosses 0 (e.g. 350,5 for Gibraltar)."""
    lon, lat = _face_lonlat_deg(mesh)
    lon = np.mod(lon, 360.0)
    lon_min, lon_max = np.mod(strait.lon_min, 360.0), np.mod(strait.lon_max, 360.0)
    if lon_min <= lon_max:
        in_lon = (lon >= lon_min) & (lon <= lon_max)
    else:
        in_lon = (lon >= lon_min) | (lon <= lon_max)
    return in_lon & (lat >= strait.lat_min) & (lat <= strait.lat_max)


def _nearest(mesh, candidates, lonlat):
    lon, lat = _face_lonlat_deg(mesh)
    xyz = lonlat_to_xyz(lon[candidates], lat[candidates])
    target = lonlat_to_xyz(*lonlat)
    return candidates[np.argmax(xyz @ target)]


def carve_strait(mesh, adjacency, landsea_mask, land_fraction, strait):
    """Return (new_mask, carved_cell_indices). If the two seeds' ocean cells
    are not connected within the strait's box, turn the cheapest path of
    land cells between them into ocean (cost of entering a land cell =
    1 + land_fraction, so the path prefers few and marginal land cells)."""
    mask = landsea_mask.copy()
    box = np.flatnonzero(_in_box(mesh, strait))
    box_ocean = box[mask[box] == 0]
    if box_ocean.size == 0:
        raise ValueError(f"strait {strait.name!r}: no ocean cell inside its box")

    a = _nearest(mesh, box_ocean, strait.seed_a)
    b = _nearest(mesh, box_ocean, strait.seed_b)

    sub_ocean = adjacency[box_ocean][:, box_ocean]
    _, labels = connected_components(sub_ocean, directed=False)
    pos = {cell: k for k, cell in enumerate(box_ocean)}
    if labels[pos[a]] == labels[pos[b]]:
        return mask, np.array([], dtype=int)

    sub = adjacency[box][:, box].tocoo()
    enter_cost = np.where(mask[box] == 0, 1e-6, 1.0 + land_fraction[box])
    weights = scipy.sparse.csr_matrix((enter_cost[sub.col], (sub.row, sub.col)), shape=sub.shape)
    box_pos = {cell: k for k, cell in enumerate(box)}
    dist, pred = dijkstra(weights, directed=True, indices=box_pos[a], return_predecessors=True)
    if not np.isfinite(dist[box_pos[b]]):
        raise ValueError(f"strait {strait.name!r}: seeds are not connected even through land inside its box")

    path = []
    k = box_pos[b]
    while k >= 0:
        path.append(box[k])
        k = pred[k]
    path = np.array(path)
    carved = path[mask[path] == 1]
    mask[carved] = 0
    return mask, carved


def fill_isolated_ocean(adjacency, landsea_mask, area):
    """Return (new_mask, filled_cell_indices): every ocean cell outside the
    largest (by area) connected ocean body becomes land."""
    mask = landsea_mask.copy()
    ocean = np.flatnonzero(mask == 0)
    if ocean.size == 0:
        return mask, np.array([], dtype=int)
    n, labels = connected_components(adjacency[ocean][:, ocean], directed=False)
    body_area = np.bincount(labels, weights=area[ocean], minlength=n)
    filled = ocean[labels != np.argmax(body_area)]
    mask[filled] = 1
    return mask, filled


def cap_north(mesh, landsea_mask, cap_lat):
    """Return (new_mask, capped_cell_indices): every cell whose center is at
    or north of `cap_lat` (geographic, degrees) becomes land. Removes the
    ocean near the pole singularity of an unrotated lat-lon ocean grid."""
    mask = landsea_mask.copy()
    _, lat = _face_lonlat_deg(mesh)
    capped = np.flatnonzero(lat >= cap_lat)
    mask[capped] = 1
    return mask, capped


def derive_fields(elevation, landsea_mask, min_depth):
    """topography and bathymetry from the regridded mean elevation, made
    consistent with the (cleaned) mask."""
    is_land = landsea_mask == 1
    topography = np.where(is_land, np.maximum(elevation, 0.0), 0.0)
    bathymetry = np.where(is_land, 0.0, np.minimum(np.minimum(elevation, 0.0), -min_depth))
    return topography, bathymetry


def clean_surface(mesh, area, elevation, land_fraction, *, land_threshold, min_depth, straits,
                  cap_lat=None, cap_elevation=0.0):
    """Apply every step-3 rule. Returns (fields dict, report dict).

    With `cap_lat` set, cells at or north of it become flat land:
    land_fraction 1 and elevation `cap_elevation` (m)."""
    adjacency = face_adjacency(mesh)
    mask = (land_fraction >= land_threshold).astype(np.int32)
    land_fraction = land_fraction.copy()
    elevation = elevation.copy()
    report = {}

    if cap_lat is not None:
        mask, capped = cap_north(mesh, mask, cap_lat)
        land_fraction[capped] = 1.0
        elevation[capped] = cap_elevation
        report["capped:north"] = capped

    """ commented out  by Tien-Yiao
    for strait in straits:
        mask, carved = carve_strait(mesh, adjacency, mask, land_fraction, strait)
        land_fraction[carved] = 0.0
        report[f"carved:{strait.name}"] = carved
    """

    mask, filled = fill_isolated_ocean(adjacency, mask, area)
    #commented out by Tien-Yiao
    #land_fraction[filled] = 1.0
    report["filled:isolated_ocean"] = filled

    topography, bathymetry = derive_fields(elevation, mask, min_depth)
    fields = {
        "land_fraction": land_fraction,
        "landsea_mask": mask,
        "elevation": elevation,
        "topography": topography,
        "bathymetry": bathymetry,
    }
    return fields, report
