Here the earth topography and bathymetry are preprocessed from an ultra high-resolution data
GEBCO (https://www.gebco.net/data-products/gridded-bathymetry-data).

## Part 1: Prepare 0.1 degree data

1. Download the raw GEBCO_2026 sub-ice grid (netCDF, ~7.4GB) and place it at the repo root as
   `GEBCO_2026_sub_ice.nc`. This file is too big for git and is not tracked.

2. Block-average it to a common 0.1 degree grid, run from the repo root:

```
python preprocessing_workflow/01_generate_bathymetry_0p1deg.py              # ~6 min -> GEBCO_bathymetry_res-0p1.nc
```

   The resulting `GEBCO_bathymetry_res-0p1.nc` (~19MB, with `elevation` and `land_fraction`)
   is the input for the rest of the workflow.

## Part 2: Regrid to user-specified grid

1. Generate regrid weights:

   a. 0.1 degree resolution -(conservative)-> surface grid (ocn/lnd) regrid weight
   b. surface grid  -> atm grid
   c. atm grid      -> surface grid

2. Regrid 0.1 degree `elevation` and `land_fraction` to surface grid (conservative).
3. Scrutinize the topography and the landsea mask on surface grid. For example, no
   isolated ocean; need throughflow that connects Indian-Pacific Basin in the
   tropics. Keep the fields consistent with `landsea_mask`:

   - where `landsea_mask = 0`: `bathymetry <= -min_depth` (configurable minimum ocean depth)
   - where `landsea_mask = 1`: `bathymetry = 0`

4. The scrutinized data is a NetCDF file with the following variables:

   a. `land_fraction` ( land fraction between 0 and 1 )
   b. `landsea_mask` ( binary land sea mask. 0 = ocean, 1 = land )
   c. `elevation` ( conservatively regridded mean elevation in meters. Positive above sea level, negative below )
   d. `topography` ( `max(elevation, 0)`. Surface height that atmosphere sees. )
   e. `bathymetry` ( `min(elevation, 0)`. Depth that ocean sees; at least `min_depth` where `landsea_mask = 0`. )

5. Regrid the clean topography and landsea mask to atm grid.

