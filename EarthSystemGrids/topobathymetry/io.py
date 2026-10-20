"""
Reading grids and writing per-cell fields for the topobathymetry workflow.
Fields are flat arrays in the mesh's face order, which is also SCRIP order
for meshes written by `StructuredQuadMesh.write_to_SCRIP_grid_file`.
"""

import numpy as np
import xarray as xr

from EarthSystemGrids.base import StructuredQuadMesh, UnstructuredGridMesh


def read_mesh(scrip_path):
    """Mesh from a SCRIP file: a StructuredQuadMesh (shape (nj, ni)) for a
    rank-2 quadrilateral grid such as those this package writes, otherwise an
    UnstructuredGridMesh (shape (nface,))."""
    with xr.open_dataset(scrip_path) as ds:
        structured = ds.sizes.get("grid_rank") == 2 and ds.sizes.get("grid_corners") == 4
    if structured:
        return StructuredQuadMesh.from_SCRIP_file(scrip_path)
    return UnstructuredGridMesh.from_SCRIP_file(scrip_path)


def global_mean(field, area):
    return float(np.sum(field * area) / np.sum(area))


def write_grid_fields(path, fields, mesh, attrs=None):
    """Write flat face-order fields to netCDF, reshaped to (nj, ni) for a
    2D mesh or kept as (ncol,) otherwise, with cell-center lat/lon.
    `fields` maps name -> (values, var_attrs)."""
    if len(mesh.shape) == 2:
        dims, shape = ("nj", "ni"), mesh.shape
    else:
        dims, shape = ("ncol",), (int(np.prod(mesh.shape)),)

    coords = {
        "lat": (dims, np.rad2deg(mesh.face_lat).reshape(shape), {"standard_name": "latitude", "units": "degrees_north"}),
        "lon": (dims, np.rad2deg(mesh.face_lon).reshape(shape), {"standard_name": "longitude", "units": "degrees_east"}),
    }
    data_vars = {}
    for name, (values, var_attrs) in fields.items():
        data_vars[name] = (dims, np.asarray(values).reshape(shape), var_attrs)

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs or {})
    ds.to_netcdf(path)


def read_grid_fields(path, names):
    """Inverse of `write_grid_fields`: flat face-order arrays."""
    with xr.open_dataset(path) as ds:
        return {name: ds[name].values.reshape(-1) for name in names}
