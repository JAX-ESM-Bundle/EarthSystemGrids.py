"""
GEBCO reading and the 0.1deg intermediate grid.

The 0.1deg grid uses the "ERA5-style" node-centered global convention:
latitude runs 90 .. -90 inclusive of both poles, longitude runs
0 .. (360 - res) inclusive, with uniform spacing `res` in degrees. A node's
grid cell is a full `res` x `res` box centered on it, except at the poles
where there is no room for a full cell beyond +-90 and the cell is a
half-width polar cap instead.
"""

import numpy as np
import xarray as xr

from EarthSystemGrids.GaussianLatLon import equally_spaced_bounds, generate_mesh, offset_equally_spaced_bounds

RES = 0.1


def node_centered_grid(res):
    """Latitude (90 -> -90) and longitude (0 -> 360-res) node coordinates
    for the ERA5-style global grid at `res` degrees spacing."""
    n_lat = round(180 / res) + 1
    n_lon = round(360 / res)
    lat = 90.0 - np.arange(n_lat) * res
    lon = np.arange(n_lon) * res
    return lat, lon


def open_gebco(path, chunk_lat, chunk_lon):
    ds = xr.open_dataset(path, chunks={"lat": chunk_lat, "lon": chunk_lon})
    native_res_lat = float(ds["lat"][1] - ds["lat"][0])
    native_res_lon = float(ds["lon"][1] - ds["lon"][0])
    if not np.isclose(native_res_lat, native_res_lon):
        raise ValueError(f"expected uniform native lat/lon spacing; got {native_res_lat} vs {native_res_lon}")
    return ds, native_res_lat


def coarsen_gebco_to_0p1deg(path, chunk_lat, chunk_lon):
    """Block-average a native GEBCO file onto the 0.1deg node-centered grid.
    Returns a Dataset with `elevation` (mean, m) and `land_fraction`
    (fraction of native pixels with elevation >= 0) on (latitude, longitude)."""
    from dask.diagnostics import ProgressBar

    ds, native_res = open_gebco(path, chunk_lat, chunk_lon)

    factor = round(RES / native_res)
    if factor % 2:
        raise ValueError(
            f"block factor {factor} (target {RES}deg / native {native_res}deg) "
            "must be even for the pole cells to split into equal half-caps"
        )
    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    if n_lon % factor or (n_lat - factor) % factor:
        raise ValueError(
            f"native grid ({n_lat}x{n_lon}) does not evenly tile into block factor {factor} "
            "with a half-block polar cap at each end"
        )
    if chunk_lon % factor:
        raise ValueError(f"chunk_lon ({chunk_lon}) must be a multiple of the block factor ({factor})")

    target_lat, target_lon = node_centered_grid(RES)
    print(f"Native {native_res:.6f}deg -> target {RES}deg (block factor {factor}) ...")

    # Roll the raw longitude axis so a *cell boundary* (not a cell center)
    # lands at the array's wrap-around point, matching the target grid's
    # first cell boundary at -RES/2, so a plain non-periodic
    # block-mean can be used afterward.
    lon_native = ds["lon"].values
    target_edge = target_lon[0] - RES / 2
    if target_edge > 180:
        target_edge -= 360
    roll_idx = int(np.argmin(np.abs(lon_native - target_edge)))

    elevation = ds["elevation"].astype(np.float32)
    is_land = (elevation >= 0).astype(np.float32)
    src = xr.Dataset({"elevation": elevation, "land_fraction": is_land})
    rolled = src.roll(lon=-roll_idx, roll_coords=False)

    print("Averaging over longitude (single pass over ~7GB; may take a while) ...")
    with ProgressBar():
        lon_coarse = rolled.coarsen(lon=factor, boundary="exact").mean().compute()

    # From here the arrays are small (n_lat x n_lon_target floats) -- do the
    # latitude pole/interior handling with plain in-memory xarray/numpy.
    pole_w = factor // 2
    south_pole = lon_coarse.isel(lat=slice(0, pole_w)).mean("lat").expand_dims(lat=1)
    north_pole = lon_coarse.isel(lat=slice(n_lat - pole_w, n_lat)).mean("lat").expand_dims(lat=1)
    interior = lon_coarse.isel(lat=slice(pole_w, n_lat - pole_w))
    interior_coarse = interior.coarsen(lat=factor, boundary="exact").mean()
    interior_desc = interior_coarse.isel(lat=slice(None, None, -1))  # native is ascending; target lat is descending

    pieces = [north_pole, interior_desc, south_pole]
    pieces = [p.drop_vars("lat", errors="ignore") for p in pieces]

    out = xr.concat(pieces, dim="lat", coords="minimal")
    out = out.rename(lat="latitude", lon="longitude")
    out = out.assign_coords(latitude=target_lat, longitude=target_lon)

    out["elevation"].attrs = {
        "standard_name": "height",
        "long_name": (
            "block-averaged elevation relative to sea level (GEBCO_2026 "
            "sub-ice, positive up, negative below sea level)"
        ),
        "units": "m",
        "cell_methods": f"lat: lon: mean (full {factor}x{factor} native cell; {pole_w}-row half-cap at each pole)",
    }
    out["land_fraction"].attrs = {
        "long_name": "fraction of native GEBCO pixels in the cell with elevation >= 0",
        "units": "1",
        "cell_methods": f"lat: lon: mean (full {factor}x{factor} native cell; {pole_w}-row half-cap at each pole)",
    }
    out["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    out.attrs = {
        "title": f"GEBCO_2026 sub-ice topography/bathymetry, block-averaged onto a {RES}deg node-centered grid",
        "source": path,
        "comment": "Part 1 intermediate product -- see workflows/topobathymetry/README.md",
    }
    return out


def build_0p1deg_mesh():
    """StructuredQuadMesh for the 0.1deg node-centered grid, with the
    half-width polar caps as real cell bounds (not bounds re-derived from
    the centers). Faces are row-major over (lat ascending, lon) -- see
    `to_mesh_order` for flattening data to match."""
    n_lat = round(180 / RES) + 1
    n_lon = round(360 / RES)
    return generate_mesh(
        offset_equally_spaced_bounds(n_lat, -90.0, 90.0),
        equally_spaced_bounds(n_lon, -RES / 2, 360.0 - RES / 2),
    )


def to_mesh_order(da):
    """Flatten a 0.1deg (latitude, longitude) field -- latitude descending
    90 -> -90 -- into `build_0p1deg_mesh`'s face order."""
    expected_lat, expected_lon = node_centered_grid(RES)
    if not (np.allclose(da["latitude"], expected_lat) and np.allclose(da["longitude"], expected_lon)):
        raise ValueError(f"field is not on the expected {RES}deg node-centered grid")
    return da.transpose("latitude", "longitude").values[::-1, :].reshape(-1).astype(np.float64)
