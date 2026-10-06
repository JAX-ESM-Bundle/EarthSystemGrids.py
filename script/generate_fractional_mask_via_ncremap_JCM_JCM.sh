#!/bin/bash
#
# Regrid the ERA5 land-sea mask and geopotential onto the JCM grid,
# conservatively (area-weighted), via NCO's ncremap -- both variables in the
# same ncremap call, so they land in the same output file. The geopotential
# is then converted to topography (surface altitude) in place, by dividing
# by gravity.
#
# ERA5's `lsm` variable is already a fractional land-sea mask (0..1) at
# 0.25deg, so conservatively remapping it directly gives a fractional mask
# on the target grid -- no thresholding step needed. ncremap infers the
# source grid from ERA5_landsea_mask.nc's own lat/lon coordinates; only the
# destination grid needs an explicit SCRIP file, generated below from this
# repo's own grid code.
#
# ERA5's `z` variable is geopotential (m2 s-2), not elevation. Conservative
# remapping is linear, so it commutes with the division by gravity -- the
# script remaps `z` first and divides afterward, rather than dividing ERA5
# `z` upfront, for exactly the same result with one fewer intermediate file.
#
# Unlike the JCM_RGLL / JCM_DSP variants, this script targets the JCM grid
# alone, so there is no second grid to map to and no ESMF_RegridWeightGen
# step -- only ncremap is needed.
#
# Needs ncremap (NCO) on PATH.

set -euo pipefail
set -x

# --- inputs you may want to change ------------------------------------------
ERA5_FILE=EarthSystemGrids/data/ERA5_landsea_mask.nc     # must have variables "lsm" and "z" on lat/lon
OUTPUT_DIR=grid                                          # holds both the land-sea/topography output and the SCRIP file

JCM_RESOLUTION=31                  # spectral truncation, e.g. 31 -> T31

usage() {
    cat <<EOF
Usage: $(basename "$0") [-e ERA5_FILE] [-o OUTPUT_DIR] [-r JCM_RESOLUTION]

  -e, --era5-file FILE        ERA5 file with "lsm" and "z" on lat/lon (default: ${ERA5_FILE})
  -o, --output-dir DIR        directory for the SCRIP grid and remapped output (default: ${OUTPUT_DIR})
  -r, --jcm-resolution N      JCM spectral truncation, e.g. 31 -> T31 (default: ${JCM_RESOLUTION})
  -h, --help                  show this help message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -e|--era5-file)
            ERA5_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--jcm-resolution)
            JCM_RESOLUTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

GRAVITY=9.80665                    # m s-2, standard gravity (ERA5 convention: elevation = z / GRAVITY)

REMAP_ALGORITHM=conserve           # area-conservative: right choice for a fractional field
# -----------------------------------------------------------------------------

if [ ! -f "$ERA5_FILE" ]; then
    echo "error: $ERA5_FILE not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

jcm_scrip="${OUTPUT_DIR}/JCM_T${JCM_RESOLUTION}.SCRIP.nc"

echo "Generating JCM T${JCM_RESOLUTION} SCRIP grid ..."
python3 -c "
from EarthSystemGrids.JCMGrid import generate_JCMGrid
mesh = generate_JCMGrid(${JCM_RESOLUTION})
mesh.write_to_SCRIP_grid_file('${jcm_scrip}', flatten=True)
"

landsea_jcm="${OUTPUT_DIR}/landsea_mask_fraction_JCM_T${JCM_RESOLUTION}.nc"

echo "Remapping ERA5 land-sea mask and geopotential onto the JCM grid ..."
ncremap -a "${REMAP_ALGORITHM}" \
    -g "${jcm_scrip}" \
    -i "${ERA5_FILE}" \
    -v lsm,z \
    -o "${landsea_jcm}"

echo "Converting remapped geopotential to topography (z / gravity) ..."
python3 -c "
import xarray as xr

GRAVITY = ${GRAVITY}

path = '${landsea_jcm}'
# Load fully and close before overwriting -- xarray's netCDF4 backend
# keeps the file open lazily otherwise, and writing back to a path it
# still has open corrupts the file.
ds = xr.open_dataset(path).load()
ds.close()

topography = ds['z'] / GRAVITY
topography.attrs = {
    'standard_name': 'surface_altitude',
    'long_name': 'surface altitude (ERA5 geopotential / gravity)',
    'units': 'm',
}
ds = ds.drop_vars('z')
ds['topography'] = topography
ds.to_netcdf(path)
"

echo "Done. Fractional land-sea mask and topography written to ${landsea_jcm}"
