Here the earth topography and bathymetry are preprocessed from an ultra high-resolution data
GEBCO (https://www.gebco.net/data-products/gridded-bathymetry-data).

The scripts here are thin command-line wrappers. The logic lives in the package:
`EarthSystemGrids.topobathymetry` (`gebco`, `cleanup`, `io`) plus
`EarthSystemGrids.base.adjacency` and `EarthSystemGrids.base.weight_regrid`. Install the
package with its extra dependencies (xarray, netCDF4, dask) first:

```
pip install -e ".[topo]"
```

The scripts can then be run from any directory; inputs and outputs default to the current one.

## Part 1: Prepare 0.1 degree data

1. Download the raw GEBCO_2026 sub-ice grid (netCDF, ~7.4GB), e.g. as `GEBCO_2026_sub_ice.nc`.
   This file is too big for git and is not tracked.

2. Block-average it to a common 0.1 degree grid:

```
python workflows/topobathymetry/01_generate_bathymetry_0p1deg.py --input GEBCO_2026_sub_ice.nc   # ~6 min -> GEBCO_bathymetry_res-0p1.nc
```

   The resulting `GEBCO_bathymetry_res-0p1.nc` (~19MB, with `elevation` and `land_fraction`)
   is the input for the rest of the workflow.

## Part 2: Regrid to user-specified grid

The atmosphere and surface grids are given as SCRIP files (e.g. from
`script/generate_fractional_mask_via_ncremap*.sh`, or `run_all.py` below). Run in an environment
with `ESMF_RegridWeightGen` and `mpirun` (e.g. the `jax` conda env), from the directory holding
`GEBCO_bathymetry_res-0p1.nc` (or pass `--input-0p1`):

```
workflows/topobathymetry/run_part2.sh \
    --atm-scrip grid/JCM_T31.SCRIP.nc \
    --sfc-scrip grid/RotatedGaussianLatLon.SCRIP.nc \
    -o part2_output --nproc 16 \
    -- --land-threshold 0.5 --min-depth 10     # optional, passed to 04_ensure_featrues_of_topobathymetry_in_surface_grid.py
```

To build the grids from this repo's grid code and run everything in one go, describe the run in a
YAML file (start from `ESG_config.yaml`: grid types and sizes, land threshold, minimum depth,
required straits) and run:

```
python workflows/topobathymetry/run_all.py --config my_run.yaml
```

This writes `sfc_topo_<sfc>.nc` (step 4) and `atm_topo_<atm>.nc` (step 5), plus the weight
files, to the output directory. Each step can also be run on its own:

| Step | Script |
|------|--------|
| 1    | `02_generate_weights.sh` |
| 2    | `03_regrid_to_surface_grid.py` |
| 3, 4 | `04_ensure_featrues_of_topobathymetry_in_surface_grid.py` (rules live in `EarthSystemGrids/topobathymetry/cleanup.py`) |
| 5    | `05_regrid_to_atm.py` |

1. Generate regrid weights:

   a. 0.1 degree resolution -(conservative)-> surface grid (ocn/lnd) regrid weight
   b. surface grid  -> atm grid
   c. atm grid      -> surface grid

2. Regrid 0.1 degree `elevation` and `land_fraction` to surface grid (conservative).
3. Scrutinize the topography and the landsea mask on surface grid. For example, no
   isolated ocean; need throughflow that connects Indian-Pacific Basin in the
   tropics. Rules, in order:

   - `landsea_mask = land_fraction >= land_threshold` (default 0.5)
   - each required strait (`--strait`, default: Indonesian throughflow) is carved open
     through the cheapest land cells inside its lon/lat box if its two basins are not connected
   - every ocean body except the largest becomes land
   - cells flipped by the two rules above get `land_fraction` 0 or 1; all others keep their
     regridded value

   Keep the fields consistent with `landsea_mask`:

   - where `landsea_mask = 0`: `bathymetry <= -min_depth` (configurable minimum ocean depth), `topography = 0`
   - where `landsea_mask = 1`: `bathymetry = 0`

4. The scrutinized data is a NetCDF file with the following variables:

   a. `land_fraction` ( land fraction between 0 and 1 )
   b. `landsea_mask` ( binary land sea mask. 0 = ocean, 1 = land )
   c. `elevation` ( conservatively regridded mean elevation in meters. Positive above sea level, negative below )
   d. `topography` ( `max(elevation, 0)`. Surface height that atmosphere sees; 0 where `landsea_mask = 0`. )
   e. `bathymetry` ( `min(elevation, 0)`. Depth that ocean sees; at least `min_depth` where `landsea_mask = 0`. )

5. Regrid the clean `land_fraction` and `topography` to atm grid (conservative), and set the atm
   `landsea_mask = land_fraction >= land_threshold`.

