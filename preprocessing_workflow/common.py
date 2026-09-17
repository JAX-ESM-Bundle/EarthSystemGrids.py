"""
Shared helpers for the GEBCO preprocessing scripts (see README.md here).

Part 1 (01_generate_bathymetry_0p1deg.py) uses the "ERA5-style"
node-centered global grid convention: latitude runs 90 .. -90 inclusive of
both poles, longitude runs 0 .. (360 - res) inclusive, with uniform spacing
`res` in degrees. A node's grid cell is a full `res` x `res` box centered on
it, except at the poles where there is no room for a full cell beyond +-90
and the cell is a half-width polar cap instead.

Part 2 (02..05) works on arbitrary SCRIP grids, as flat `(grid_size,)`
arrays in SCRIP order, and applies ESMF_RegridWeightGen weight files to
them.
"""

import numpy as np
import scipy.sparse
import xarray as xr

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
DEFAULT_GEBCO_FILE = str(REPO_ROOT / "GEBCO_2026_sub_ice.nc")

SOURCE_RES = 0.1


def node_centered_grid(res):
    """Latitude (90 -> -90) and longitude (0 -> 360-res) node coordinates
    for the ERA5-style global grid at `res` degrees spacing."""
    n_lat = round(180 / res) + 1
    n_lon = round(360 / res)
    lat = 90.0 - np.arange(n_lat) * res
    lon = np.arange(n_lon) * res
    return lat, lon


def node_centered_lat_edges(res):
    """Ascending cell-edge array (length n_lat+1) for the latitude grid at
    `res` degrees: interior edges are node midpoints, the outer edges are
    the poles (+-90), giving interior nodes a full-width cell and the pole
    nodes a half-width cap."""
    lat, _ = node_centered_grid(res)
    lat_asc = lat[::-1]  # -90 -> 90
    interior_edges = (lat_asc[:-1] + lat_asc[1:]) / 2
    return np.concatenate([[-90.0], interior_edges, [90.0]])


def node_centered_lon_edges(res):
    """Ascending cell-edge array (length n_lon+1) for the longitude grid at
    `res` degrees: every node (0 .. 360-res) gets a full-width cell, so
    edges run -res/2 .. 360-res/2."""
    _, lon = node_centered_grid(res)
    return np.concatenate([lon - res / 2, [lon[-1] + res / 2]])


def open_gebco(path, chunk_lat, chunk_lon):
    ds = xr.open_dataset(path, chunks={"lat": chunk_lat, "lon": chunk_lon})
    native_res_lat = float(ds["lat"][1] - ds["lat"][0])
    native_res_lon = float(ds["lon"][1] - ds["lon"][0])
    if not np.isclose(native_res_lat, native_res_lon):
        raise ValueError(f"expected uniform native lat/lon spacing; got {native_res_lat} vs {native_res_lon}")
    return ds, native_res_lat


# --- Part 2 ------------------------------------------------------------------


def build_0p1deg_mesh():
    """StructuredQuadMesh for Part 1's 0.1deg node-centered grid, with the
    half-width polar caps as real cell bounds (not bounds re-derived from
    the centers). Faces are row-major over (lat ascending, lon) -- see
    `to_mesh_order` for flattening data to match."""
    from EarthSystemGrids.GaussianLatLon import generate_mesh

    return generate_mesh(node_centered_lat_edges(SOURCE_RES), node_centered_lon_edges(SOURCE_RES))


def to_mesh_order(da):
    """Flatten a Part 1 (latitude, longitude) field -- latitude descending
    90 -> -90 -- into `build_0p1deg_mesh`'s face order."""
    expected_lat, expected_lon = node_centered_grid(SOURCE_RES)
    if not (np.allclose(da["latitude"], expected_lat) and np.allclose(da["longitude"], expected_lon)):
        raise ValueError(f"field is not on the expected {SOURCE_RES}deg node-centered grid")
    return da.transpose("latitude", "longitude").values[::-1, :].reshape(-1).astype(np.float64)


def _to_degrees(da):
    return np.rad2deg(da.values) if da.attrs.get("units", "degrees").startswith("rad") else da.values


def read_scrip(path):
    """Grid description from a SCRIP file, all in degrees, flat SCRIP order:
    dict with center_lon/center_lat (grid_size,), corner_lon/corner_lat
    (grid_size, grid_corners), and grid_dims (as stored: [ni, nj] for a
    rank-2 grid)."""
    with xr.open_dataset(path) as ds:
        return {
            "center_lon": _to_degrees(ds["grid_center_lon"]).reshape(-1),
            "center_lat": _to_degrees(ds["grid_center_lat"]).reshape(-1),
            "corner_lon": _to_degrees(ds["grid_corner_lon"]).reshape(-1, ds.sizes["grid_corners"]),
            "corner_lat": _to_degrees(ds["grid_corner_lat"]).reshape(-1, ds.sizes["grid_corners"]),
            "grid_dims": [int(n) for n in ds["grid_dims"].values],
        }


class WeightRegridder:
    """Applies an ESMF_RegridWeightGen weight file to flat (n_a,) arrays,
    in float64 (scipy.sparse). Destination cells no source cell covers
    (frac_b == 0) come back NaN."""

    def __init__(self, weight_file):
        with xr.open_dataset(weight_file) as ds:
            self.n_a = ds.sizes["n_a"]
            self.n_b = ds.sizes["n_b"]
            self.matrix = scipy.sparse.csr_matrix(
                (ds["S"].values, (ds["row"].values - 1, ds["col"].values - 1)),
                shape=(self.n_b, self.n_a),
            )
            self.frac_b = ds["frac_b"].values
            self.area_a = ds["area_a"].values
            self.area_b = ds["area_b"].values
        self.weight_file = str(weight_file)

    def __call__(self, field_a):
        field_a = np.asarray(field_a, dtype=np.float64)
        if field_a.shape != (self.n_a,):
            raise ValueError(f"expected a flat ({self.n_a},) field for {self.weight_file}, got {field_a.shape}")
        field_b = self.matrix @ field_a
        field_b[self.frac_b <= 0] = np.nan
        return field_b


def global_mean(field, area):
    return float(np.sum(field * area) / np.sum(area))


def write_grid_fields(path, fields, scrip_path, attrs=None):
    """Write flat SCRIP-order fields to netCDF, reshaped to (nj, ni) for a
    rank-2 grid (grid_dims = [ni, nj], i fastest -- i.e. C order over
    (nj, ni)) or kept as (ncol,) otherwise, with cell-center lat/lon."""
    grid = read_scrip(scrip_path)
    dims_scrip = grid["grid_dims"]
    if len(dims_scrip) == 2:
        dims, shape = ("nj", "ni"), (dims_scrip[1], dims_scrip[0])
    else:
        dims, shape = ("ncol",), (int(np.prod(dims_scrip)),)

    coords = {
        "lat": (dims, grid["center_lat"].reshape(shape), {"standard_name": "latitude", "units": "degrees_north"}),
        "lon": (dims, grid["center_lon"].reshape(shape), {"standard_name": "longitude", "units": "degrees_east"}),
    }
    data_vars = {}
    for name, (values, var_attrs) in fields.items():
        data_vars[name] = (dims, np.asarray(values).reshape(shape), var_attrs)

    ds = xr.Dataset(data_vars, coords=coords, attrs={"grid_file": str(scrip_path), **(attrs or {})})
    ds.to_netcdf(path)


def read_grid_fields(path, names):
    """Inverse of `write_grid_fields`: flat SCRIP-order arrays."""
    with xr.open_dataset(path) as ds:
        return {name: ds[name].values.reshape(-1) for name in names}


def lonlat_to_xyz(lon_deg, lat_deg):
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def face_adjacency(grid, tol=1e-9):
    """Symmetric sparse (nface, nface) graph: two faces are neighbours when
    they share an edge. Shared edges are found by coordinates (edge
    midpoints on the unit sphere coincide), not by node ids, so faces on
    either side of a longitude seam -- whose nodes this repo's meshes do not
    merge -- are still connected. Degenerate edges (both corners on the same
    point, e.g. a pole) connect nothing."""
    corners = lonlat_to_xyz(grid["corner_lon"], grid["corner_lat"])  # (nface, nc, 3)
    nface, nc, _ = corners.shape
    a = corners
    b = np.roll(corners, -1, axis=1)
    midpoints = (0.5 * (a + b)).reshape(-1, 3)
    valid = (np.linalg.norm(a - b, axis=-1) > tol).reshape(-1)
    edge_face = np.repeat(np.arange(nface), nc)

    idx = np.flatnonzero(valid)
    from scipy.spatial import cKDTree

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
