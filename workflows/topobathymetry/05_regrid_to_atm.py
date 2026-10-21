"""
Part 2, step 5: conservatively regrid the cleaned surface `land_fraction`
and `topography` (from 04_ensure_featrues_of_topobathymetry_in_surface_grid.py) onto the atmosphere grid, and
threshold the atm land fraction into a binary `landsea_mask` with the same
threshold used on the surface grid.
"""

import argparse

import numpy as np
import xarray as xr

from EarthSystemGrids.base.weight_regrid import WeightRegridder
from EarthSystemGrids.topobathymetry.io import global_mean, read_grid_fields, read_mesh, write_grid_fields

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="output of 04_ensure_featrues_of_topobathymetry_in_surface_grid.py")
    parser.add_argument("--weight", required=True, help="conserve weight file, surface grid -> atm grid")
    parser.add_argument("--atm-scrip", required=True, help="SCRIP file of the atmosphere grid")
    parser.add_argument("--output", required=True, help="output netCDF path")
    parser.add_argument(
        "--land-threshold",
        type=float,
        default=None,
        help="land if land_fraction >= this (default: the threshold stored by 04_ensure_featrues_of_topobathymetry_in_surface_grid.py)",
    )
    args = parser.parse_args()

    threshold = args.land_threshold
    if threshold is None:
        with xr.open_dataset(args.input) as ds:
            threshold = float(ds.attrs["land_threshold"])

    regrid = WeightRegridder(args.weight)
    sfc = read_grid_fields(args.input, ["land_fraction", "topography"])

    atm = {}
    for name, values in sfc.items():
        out = regrid(values)
        n_missing = int((out != out).sum())
        if n_missing:
            raise ValueError(f"{n_missing} atm cells have no surface-grid source coverage")
        atm[name] = out

    print("Area-weighted global land fraction:")
    print(f"  surface: {global_mean(sfc['land_fraction'], regrid.area_a):.8f}")
    print(f"  atm    : {global_mean(atm['land_fraction'], regrid.area_b):.8f}")

    print("Write to:", args.output)
    write_grid_fields(
        args.output,
        {
            "land_fraction": (atm["land_fraction"], {
                "long_name": "fraction of the cell covered by land",
                "units": "1",
            }),
            "landsea_mask": ((atm["land_fraction"] >= threshold).astype(np.int32), {
                "long_name": "binary land-sea mask",
                "flag_values": "0 1",
                "flag_meanings": "ocean land",
            }),
            "topography": (atm["topography"], {
                "long_name": "surface height seen by the atmosphere (0 over ocean)",
                "units": "m",
            }),
        },
        read_mesh(args.atm_scrip),
        attrs={"grid_file": args.atm_scrip, "source": args.input, "weight_file": args.weight, "land_threshold": threshold},
    )
