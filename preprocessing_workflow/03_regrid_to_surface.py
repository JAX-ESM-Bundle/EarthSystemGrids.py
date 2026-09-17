"""
Part 2, step 2: conservatively regrid Part 1's 0.1deg `elevation` and
`land_fraction` onto the surface grid, using the GEBCO_0p1 -> surface
weights from 02_generate_weights.sh.

Fails if any surface cell is not covered by the 0.1deg grid (the grid must
be global), and prints the area-weighted global land fraction on both grids
as a conservation check.
"""

import argparse

import xarray as xr

from common import WeightRegridder, global_mean, to_mesh_order, write_grid_fields

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="preprocessing_workflow/GEBCO_bathymetry_res-0p1.nc", help="Part 1 output")
    parser.add_argument("--weight", required=True, help="conserve weight file, GEBCO_0p1 -> surface grid")
    parser.add_argument("--sfc-scrip", required=True, help="SCRIP file of the surface grid")
    parser.add_argument("--output", required=True, help="output netCDF path")
    args = parser.parse_args()

    regrid = WeightRegridder(args.weight)

    with xr.open_dataset(args.input) as ds:
        src = {name: to_mesh_order(ds[name]) for name in ("elevation", "land_fraction")}

    fields = {}
    for name, values in src.items():
        out = regrid(values)
        n_missing = int((out != out).sum())
        if n_missing:
            raise ValueError(f"{n_missing} surface cells have no 0.1deg source coverage")
        fields[name] = out

    print("Area-weighted global land fraction:")
    print(f"  0.1deg : {global_mean(src['land_fraction'], regrid.area_a):.8f}")
    print(f"  surface: {global_mean(fields['land_fraction'], regrid.area_b):.8f}")

    print("Write to:", args.output)
    write_grid_fields(
        args.output,
        {
            "elevation": (fields["elevation"], {
                "long_name": "conservatively regridded mean elevation (positive above sea level)",
                "units": "m",
            }),
            "land_fraction": (fields["land_fraction"], {
                "long_name": "fraction of the cell covered by land",
                "units": "1",
            }),
        },
        args.sfc_scrip,
        attrs={"source": args.input, "weight_file": args.weight},
    )
