"""
Surface-grid cleanup rules for Part 2 step 3 (used by 04_clean_surface.py).

All functions take flat SCRIP-order arrays plus a `grid` dict from
`common.read_scrip` and an adjacency graph from `common.face_adjacency`.
Masks use the README convention: 1 = land, 0 = ocean.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from common import face_adjacency, lonlat_to_xyz


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


def _in_box(grid, strait):
    """Box longitudes are taken mod 360; lon_min > lon_max means the box
    crosses 0 (e.g. 350,5 for Gibraltar)."""
    lon = np.mod(grid["center_lon"], 360.0)
    lat = grid["center_lat"]
    lon_min, lon_max = np.mod(strait.lon_min, 360.0), np.mod(strait.lon_max, 360.0)
    if lon_min <= lon_max:
        in_lon = (lon >= lon_min) & (lon <= lon_max)
    else:
        in_lon = (lon >= lon_min) | (lon <= lon_max)
    return in_lon & (lat >= strait.lat_min) & (lat <= strait.lat_max)


def _nearest(grid, candidates, lonlat):
    xyz = lonlat_to_xyz(grid["center_lon"][candidates], grid["center_lat"][candidates])
    target = lonlat_to_xyz(*lonlat)
    return candidates[np.argmax(xyz @ target)]


def carve_strait(grid, adjacency, landsea_mask, land_fraction, strait):
    """Return (new_mask, carved_cell_indices). If the two seeds' ocean cells
    are not connected within the strait's box, turn the cheapest path of
    land cells between them into ocean (cost of entering a land cell =
    1 + land_fraction, so the path prefers few and marginal land cells)."""
    mask = landsea_mask.copy()
    box = np.flatnonzero(_in_box(grid, strait))
    box_ocean = box[mask[box] == 0]
    if box_ocean.size == 0:
        raise ValueError(f"strait {strait.name!r}: no ocean cell inside its box")

    a = _nearest(grid, box_ocean, strait.seed_a)
    b = _nearest(grid, box_ocean, strait.seed_b)

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


def derive_fields(elevation, landsea_mask, min_depth):
    """topography and bathymetry from the regridded mean elevation, made
    consistent with the (cleaned) mask."""
    is_land = landsea_mask == 1
    topography = np.where(is_land, np.maximum(elevation, 0.0), 0.0)
    bathymetry = np.where(is_land, 0.0, np.minimum(np.minimum(elevation, 0.0), -min_depth))
    return topography, bathymetry


def clean_surface(grid, area, elevation, land_fraction, *, land_threshold, min_depth, straits):
    """Apply every step-3 rule. Returns (fields dict, report dict)."""
    adjacency = face_adjacency(grid)
    mask = (land_fraction >= land_threshold).astype(np.int32)
    land_fraction = land_fraction.copy()
    report = {}

    for strait in straits:
        mask, carved = carve_strait(grid, adjacency, mask, land_fraction, strait)
        land_fraction[carved] = 0.0
        report[f"carved:{strait.name}"] = carved

    mask, filled = fill_isolated_ocean(adjacency, mask, area)
    land_fraction[filled] = 1.0
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
