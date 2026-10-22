"""
Part 2, steps 3-4: clean the regridded surface fields (from
03_regrid_to_surface_grid.py) and write the final surface file with
`land_fraction`, `landsea_mask`, `elevation`, `topography`, `bathymetry`.

Rules, in order (see EarthSystemGrids/topobathymetry/cleanup.py):
  1. landsea_mask = land_fraction >= --land-threshold  (1 = land, 0 = ocean)
     with --cap-lat: cells at or north of it become flat land
     (land_fraction 1, elevation --cap-elevation)
  2. each required strait (default: Indonesian throughflow) is carved open
     if its two basins are not connected inside its box
  3. every ocean body except the largest becomes land
  4. cells flipped by 2/3 get land_fraction 0/1; all others keep their
     regridded value
  5. topography = max(elevation, 0) over land, 0 over ocean
     bathymetry = min(elevation, 0, -min_depth) over ocean, 0 over land
"""

import argparse

import numpy as np

from EarthSystemGrids.base.weight_regrid import WeightRegridder
from EarthSystemGrids.topobathymetry.cleanup import INDONESIAN_THROUGHFLOW, Strait, clean_surface
from EarthSystemGrids.topobathymetry.io import read_grid_fields, read_mesh, write_grid_fields

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="output of 03_regrid_to_surface_grid.py")
    parser.add_argument("--sfc-scrip", required=True, help="SCRIP file of the surface grid")
    parser.add_argument(
        "--area-weight",
        required=True,
        help="any conserve weight file whose destination is the surface grid (its area_b gives cell areas)",
    )
    parser.add_argument("--output", required=True, help="output netCDF path")
    parser.add_argument("--land-threshold", type=float, default=0.3, help="land if land_fraction >= this (default: 0.5)")
    parser.add_argument("--min-depth", type=float, default=10.0, help="minimum ocean depth in m (default: 10)")
    parser.add_argument(
        "--strait",
        action="append",
        default=None,
        metavar="NAME:LON_MIN,LON_MAX,LAT_MIN,LAT_MAX:LON_A,LAT_A:LON_B,LAT_B",
        help="required connection between two basins; repeatable. Default: Indonesian throughflow",
    )
    parser.add_argument("--no-straits", action="store_true", help="skip strait carving entirely")
    parser.add_argument(
        "--cap-lat",
        type=float,
        default=None,
        help="make cells at or north of this latitude (deg) flat land (default: no cap)",
    )
    parser.add_argument("--cap-elevation", type=float, default=0.0, help="elevation of capped cells in m (default: 0)")
    args = parser.parse_args()

    if args.no_straits:
        straits = []
    elif args.strait:
        straits = [Strait.parse(s) for s in args.strait]
    else:
        straits = [INDONESIAN_THROUGHFLOW]

    mesh = read_mesh(args.sfc_scrip)
    area = WeightRegridder(args.area_weight).area_b
    raw = read_grid_fields(args.input, ["elevation", "land_fraction"])

    fields, report = clean_surface(
        mesh, area, raw["elevation"], raw["land_fraction"],
        land_threshold=args.land_threshold, min_depth=args.min_depth, straits=straits,
        cap_lat=args.cap_lat, cap_elevation=args.cap_elevation,
    )

    print("Cleanup report:")
    for rule, cells in report.items():
        print(f"  {rule}: {cells.size} cells")
    mask = fields["landsea_mask"]
    print(f"  land cells: {int(mask.sum())} / {mask.size}")

    print("Write to:", args.output)
    write_grid_fields(
        args.output,
        {
            "land_fraction": (fields["land_fraction"], {
                "long_name": "fraction of the cell covered by land (0 or 1 where cleanup flipped the cell)",
                "units": "1",
            }),
            "landsea_mask": (mask.astype(np.int32), {
                "long_name": "binary land-sea mask",
                "flag_values": "0 1",
                "flag_meanings": "ocean land",
            }),
            "elevation": (fields["elevation"], {
                "long_name": "conservatively regridded mean elevation (positive above sea level)",
                "units": "m",
            }),
            "topography": (fields["topography"], {
                "long_name": "surface height seen by the atmosphere: max(elevation, 0) over land, 0 over ocean",
                "units": "m",
            }),
            "bathymetry": (fields["bathymetry"], {
                "long_name": "seafloor height seen by the ocean: min(elevation, 0, -min_depth) over ocean, 0 over land",
                "units": "m",
            }),
        },
        mesh,
        attrs={
            "grid_file": args.sfc_scrip,
            "source": args.input,
            "land_threshold": args.land_threshold,
            "min_depth": args.min_depth,
            "straits": "; ".join(f"{s.name} box=({s.lon_min},{s.lon_max},{s.lat_min},{s.lat_max})" for s in straits),
            **({"cap_lat": args.cap_lat, "cap_elevation": args.cap_elevation} if args.cap_lat is not None else {}),
            **{f"cells_{rule.replace(':', '_')}": int(cells.size) for rule, cells in report.items()},
        },
    )
