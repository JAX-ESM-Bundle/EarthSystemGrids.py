#!/bin/bash
#
# Build a capped-Earth land-sea mask and remap it onto two grids that are,
# by construction, the SAME grid expressed two ways:
#
#   1. the JCM T31 spectral grid, and
#   2. a RotatedGaussianLatLon grid with zero rotation, Gaussian latitude
#      spacing, and its first longitude cell centered on 0 deg.
#
# The RGLL settings below are what make the two coincide. A spectral grid's
# rows are Gaussian quadrature latitudes -- non-uniform, compressing towards
# the poles -- so --rgll-lat-spacing gaussian is required; the default
# equally-spaced faces would give a different grid with the same cell count.
# Likewise a spectral grid centers its first longitude cell on 0 deg rather
# than starting its first FACE there, hence --rgll-lon-centered. With
# nlat=48, nlon=96 (T31's Gaussian grid) and no rotation, the resulting mesh
# reproduces JCM T31's centers, corners and cell areas to machine precision.
#
# Weights are then generated between the two, in both directions and for
# both methods. They are redundant by construction -- the grids are
# identical, so the maps are the identity -- and generated deliberately, so
# downstream code has a weight file for every (source, destination) pair
# without special-casing the ones that happen to coincide. Each per-grid run
# below also emits its own self-to-self weights, so the full matrix is
# present.

set -euo pipefail

OUTPUT_DIR=capped_ERA5
CAPPED_MASK=capped_ERA5_landsea_mask.nc

JCM_RESOLUTION=31
RGLL_NLAT=48                       # T31's Gaussian grid is 96 x 48 ...
RGLL_NLON=96                       # ... so these must track JCM_RESOLUTION

WEIGHT_METHODS=(bilinear conserve)

jcm_label="JCM_T${JCM_RESOLUTION}"
rgll_label="RotatedGaussianLatLon"

python generate_rotated_grid_capped_mask.py \
    --capped-latitude 85.0 \
    --rotation-deg 0.0 \
    --axis-longitude-deg 0.0 \
    --output "${CAPPED_MASK}"

./script/generate_fractional_mask_via_ncremap.sh \
    --target-grid JCM \
    --era5-file "${CAPPED_MASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --jcm-resolution "${JCM_RESOLUTION}"

./script/generate_fractional_mask_via_ncremap.sh \
    --target-grid RGLL \
    --era5-file "${CAPPED_MASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --rgll-nlat "${RGLL_NLAT}" \
    --rgll-nlon "${RGLL_NLON}" \
    --rgll-lat-spacing gaussian \
    --rgll-lon-centered \
    --rgll-rotation-deg 0.0 \
    --rgll-axis-longitude-deg 0.0

echo "Generating ${jcm_label} <-> ${rgll_label} regridding weights ..."
for method in "${WEIGHT_METHODS[@]}"; do
    ESMF_RegridWeightGen \
        -s "${OUTPUT_DIR}/${jcm_label}.SCRIP.nc" -d "${OUTPUT_DIR}/${rgll_label}.SCRIP.nc" \
        -m "${method}" \
        -w "${OUTPUT_DIR}/weight_algo-${method}_${jcm_label}_to_${rgll_label}.nc"
    ESMF_RegridWeightGen \
        -s "${OUTPUT_DIR}/${rgll_label}.SCRIP.nc" -d "${OUTPUT_DIR}/${jcm_label}.SCRIP.nc" \
        -m "${method}" \
        -w "${OUTPUT_DIR}/weight_algo-${method}_${rgll_label}_to_${jcm_label}.nc"
done

echo "Done. Grids, masks and weights written to ${OUTPUT_DIR}/"
