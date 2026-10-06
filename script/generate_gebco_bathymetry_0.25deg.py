"""
Fetch the GEBCO_2026 sub-ice topography/bathymetry grid over OPeNDAP and
block-average it onto the exact same lat/lon grid as
EarthSystemGrids/data/ERA5_landsea_mask.nc, as a high-resolution-derived
complement to it.

GEBCO_2026 sub-ice is a pixel-registered (cell-centered) grid at 15
arc-second (1/240deg) resolution -- 43200 x 86400 points, ~7GB of int16
`elevation` (m, positive up, negative below sea level).

ERA5's grid is node-centered: 721 latitudes running 90 .. -90 inclusive of
both poles, and 1440 longitudes running 0 .. 359.75, each grid point being
a sample location rather than a cell-average location. To average GEBCO
onto this grid faithfully:
  - Latitude: interior ERA5 nodes get a full 0.25deg cell (60 native rows);
    the +-90 pole nodes get a half-width polar cap (30 native rows), since
    there's no room for a full cell beyond the pole.
  - Longitude: periodic, so every ERA5 node gets a full 0.25deg cell (60
    native columns); the native array is rolled first so a cell boundary
    (not a cell center) lands on the periodic seam, avoiding the need to
    average across the array wrap-around.

Needs xarray, dask, and either netCDF4 or pydap (for the OPeNDAP request).
"""

import argparse

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

DEFAULT_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026/"
    "sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc"
)
DEFAULT_ERA5_FILE = "EarthSystemGrids/data/ERA5_landsea_mask.nc"


def coarsen_gebco_to_era5_grid(url, era5_file, chunk_lat, chunk_lon):
    era5 = xr.open_dataset(era5_file)
    era5_lat = era5["latitude"].values
    era5_lon = era5["longitude"].values

    if not (np.isclose(era5_lat[0], 90.0) and np.isclose(era5_lat[-1], -90.0)):
        raise ValueError(
            f"expected {era5_file}'s latitude to run 90 -> -90 inclusive of "
            f"both poles (node-centered grid); got {era5_lat[0]} -> {era5_lat[-1]}"
        )
    if not np.isclose(era5_lon[0], 0.0):
        raise ValueError(
            f"expected {era5_file}'s longitude to start at 0.0; got {era5_lon[0]}"
        )

    lat_spacing = float(era5_lat[0] - era5_lat[1])
    lon_spacing = float(era5_lon[1] - era5_lon[0])
    if not np.isclose(lat_spacing, lon_spacing):
        raise ValueError(f"expected uniform lat/lon spacing; got {lat_spacing} vs {lon_spacing}")
    target_res = lat_spacing

    ds = xr.open_dataset(url, chunks={"lat": chunk_lat, "lon": chunk_lon})
    native_res_lat = float(ds["lat"][1] - ds["lat"][0])
    native_res_lon = float(ds["lon"][1] - ds["lon"][0])
    if not np.isclose(native_res_lat, native_res_lon):
        raise ValueError(f"expected uniform native lat/lon spacing; got {native_res_lat} vs {native_res_lon}")

    factor = round(target_res / native_res_lat)
    if factor % 2:
        raise ValueError(
            f"block factor {factor} (target {target_res}deg / native {native_res_lat}deg) "
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

    print(
        f"Native {native_res_lat:.6f}deg -> target {target_res}deg "
        f"(block factor {factor}, longitude roll to align periodic seam) ..."
    )

    # Roll the raw (pre-coarsening) longitude axis so a *cell boundary* --
    # not a cell center -- lands at the array's wrap-around point, so a
    # plain non-periodic block-mean can be used afterward.
    lon_native = ds["lon"].values
    target_edge = era5_lon[0] - lon_spacing / 2  # == -0.125
    if target_edge > 180:
        target_edge -= 360
    roll_idx = int(np.argmin(np.abs(lon_native - target_edge)))

    elevation = ds["elevation"].astype(np.float32)
    rolled = elevation.roll(lon=-roll_idx, roll_coords=False)

    print("Downloading and averaging over longitude (this pulls ~7GB over OPeNDAP; may take a while) ...")
    with ProgressBar():
        lon_coarse = rolled.coarsen(lon=factor, boundary="exact").mean().compute()

    # From here on the array is small (n_lat x 1440 floats) -- do the
    # latitude pole/interior handling with plain in-memory xarray/numpy.
    pole_w = factor // 2
    south_pole = lon_coarse.isel(lat=slice(0, pole_w)).mean("lat").expand_dims(lat=1)
    north_pole = lon_coarse.isel(lat=slice(n_lat - pole_w, n_lat)).mean("lat").expand_dims(lat=1)
    interior = lon_coarse.isel(lat=slice(pole_w, n_lat - pole_w))
    interior_coarse = interior.coarsen(lat=factor, boundary="exact").mean()
    interior_desc = interior_coarse.isel(lat=slice(None, None, -1))  # native is ascending; ERA5 lat is descending

    # The three pieces' 'lat' coordinates (block-mean native positions, or
    # none) don't matter -- the real ERA5 labels are assigned below -- so
    # drop them here to avoid a spurious index-alignment mismatch on concat.
    pieces = [north_pole, interior_desc, south_pole]
    pieces = [p.drop_vars("lat", errors="ignore") for p in pieces]

    elevation_out = xr.concat(pieces, dim="lat", coords="minimal")
    elevation_out = elevation_out.rename(lat="latitude", lon="longitude")
    elevation_out = elevation_out.assign_coords(latitude=era5_lat, longitude=era5_lon)

    elevation_out.attrs = {
        "standard_name": "height",
        "long_name": (
            "block-averaged elevation relative to sea level (GEBCO_2026 "
            "sub-ice, positive up, negative below sea level), on the ERA5 "
            "node-centered 0.25deg grid"
        ),
        "units": "m",
        "source": url,
        "cell_methods": (
            f"lat: lon: mean (full {factor}x{factor} native cell per interior "
            f"node; {pole_w}-row half-cap at each pole)"
        ),
    }

    out = elevation_out.to_dataset(name="elevation")
    out["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    out.attrs = {
        "title": f"GEBCO_2026 sub-ice topography/bathymetry, averaged onto the ERA5 {target_res}deg grid",
        "source": url,
        "comment": f"Complement to {era5_file} -- same latitude/longitude grid points.",
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="GEBCO OPeNDAP dataset URL")
    parser.add_argument(
        "--era5-file",
        type=str,
        default=DEFAULT_ERA5_FILE,
        help="ERA5 file whose lat/lon grid the output is averaged onto",
    )
    parser.add_argument(
        "--chunk-lat",
        type=int,
        default=3600,
        help="Dask chunk size along lat for the remote read (default: 3600)",
    )
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=3600,
        help="Dask chunk size along lon for the remote read; must be a multiple of the block factor (default: 3600)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="EarthSystemGrids/data/GEBCO_2026_bathymetry_0.25deg.nc",
        help="Output netCDF path",
    )
    args = parser.parse_args()

    out = coarsen_gebco_to_era5_grid(args.url, args.era5_file, args.chunk_lat, args.chunk_lon)

    encoding = {"elevation": {"zlib": True, "complevel": 4}}
    print("Write to:", args.output)
    out.to_netcdf(args.output, encoding=encoding)
