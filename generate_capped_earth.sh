#!/bin/bash


python generate_rotated_grid_capped_mask.py \
    --capped-latitude 85.0 \
    --rotation-deg 0.0 \
    --axis-longitude-deg 0.0 \
    --output capped_ERA5_landsea_mask.nc

./script/generate_fractional_mask_via_ncremap.sh \
    --target-grid JCM \
    --era5-file capped_ERA5_landsea_mask.nc \
    --output-dir capped_ERA5 \
    --jcm-resolution 31

