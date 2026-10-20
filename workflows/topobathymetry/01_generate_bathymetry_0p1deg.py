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

from EarthSystemGrids.topobathymetry.gebco import coarsen_gebco_to_0p1deg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the native GEBCO netCDF file (e.g. GEBCO_2026_sub_ice.nc)")
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
