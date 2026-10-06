"""
Part 1, step 1: block-average the native GEBCO_2026 sub-ice topography/
bathymetry grid (15 arc-second / 1/240deg, 43200 x 86400, ~7.4GB) down onto
a common 0.1deg ERA5-style node-centered global grid, so downstream steps
work with a ~50MB file instead of the raw high-resolution binary (which
does not belong in git -- see the README in this directory).

For each 0.1deg cell, two fields are written:
  - elevation:     mean native elevation (m, positive up / negative below
                    sea level) -- the bathymetry field itself.
  - land_fraction: fraction of native pixels in the cell with
                    elevation >= 0 -- kept alongside the mean elevation so
                    Part 2's conservative regrid can produce an accurate
                    fractional land-sea mask instead of one derived from a
                    threshold on the already-averaged elevation.

Native resolution (1/240deg) divides 0.1deg exactly (factor 24), so this
step is a plain block mean; arbitrary target grids are handled by Part 2's
ESMF conservative regrid instead.
"""

import argparse

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from common import DEFAULT_GEBCO_FILE, node_centered_grid, open_gebco

TARGET_RES = 0.1


def coarsen_gebco_to_0p1deg(path, chunk_lat, chunk_lon):
    ds, native_res = open_gebco(path, chunk_lat, chunk_lon)

    factor = round(TARGET_RES / native_res)
    if factor % 2:
        raise ValueError(
            f"block factor {factor} (target {TARGET_RES}deg / native {native_res}deg) "
            "must be even for the pole cells to split into equal half-caps"
        )
    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    if n_lon % factor or (n_lat - factor) % factor:
        raise ValueError(
            f"native grid ({n_lat}x{n_lon}) does not evenly tile into block factor {factor} "
            "with a half-block polar cap at each end"
        )
    if chunk_lon % factor:
        raise ValueError(f"--chunk-lon ({chunk_lon}) must be a multiple of the block factor ({factor})")

    target_lat, target_lon = node_centered_grid(TARGET_RES)
    print(f"Native {native_res:.6f}deg -> target {TARGET_RES}deg (block factor {factor}) ...")

    # Roll the raw longitude axis so a *cell boundary* (not a cell center)
    # lands at the array's wrap-around point, matching the target grid's
    # first cell boundary at -TARGET_RES/2, so a plain non-periodic
    # block-mean can be used afterward.
    lon_native = ds["lon"].values
    target_edge = target_lon[0] - TARGET_RES / 2
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
        "title": f"GEBCO_2026 sub-ice topography/bathymetry, block-averaged onto a {TARGET_RES}deg node-centered grid",
        "source": path,
        "comment": "Part 1 intermediate product -- see preprocessing_workflow/README.md",
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=str, default=DEFAULT_GEBCO_FILE, help="Path to the native GEBCO netCDF file")
    parser.add_argument("--chunk-lat", type=int, default=3600, help="Dask chunk size along lat (default: 3600)")
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=3600,
        help="Dask chunk size along lon; must be a multiple of the block factor (default: 3600)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="GEBCO_bathymetry_res-0p1.nc",
        help="Output netCDF path",
    )
    args = parser.parse_args()

    out = coarsen_gebco_to_0p1deg(args.input, args.chunk_lat, args.chunk_lon)

    encoding = {"elevation": {"zlib": True, "complevel": 4}, "land_fraction": {"zlib": True, "complevel": 4}}
    print("Write to:", args.output)
    out.to_netcdf(args.output, encoding=encoding)
