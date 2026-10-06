#!/bin/bash
#
# Run all of Part 2 with no pre-built inputs: generate the atm grid (JCM
# spectral grid) and the surface grid (rotated Gaussian lat-lon grid) as
# SCRIP files from this repo's grid code, then hand them to run_part2.sh.
#
# The grid options mirror script/generate_fractional_mask_via_ncremap.sh.
# Needs Part 1's GEBCO_bathymetry_res-0p1.nc, and ESMF_RegridWeightGen +
# mpirun on PATH (e.g. the `jax` conda env). Run from the repo root.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --- inputs you may want to change ------------------------------------------
OUTPUT_DIR=part2_output
NPROC=16

JCM_RESOLUTION=31                  # spectral truncation, e.g. 31 -> T31

RGLL_NLAT=60
RGLL_NLON=120
RGLL_LAT_SPACING=equal             # equal | offset | gaussian
RGLL_LON_CENTERED=0                # 1 -> first longitude cell centered on 0 deg
RGLL_ROTATION_DEG=12.0             # 0 -> no rotation
RGLL_ROTATION_AXIS_LONGITUDE_DEG=48.0

LAND_THRESHOLD=0.3
MIN_DEPTH=10.0

# Required straits for 04_clean_surface.py
# (NAME:LON_MIN,LON_MAX,LAT_MIN,LAT_MAX:LON_A,LAT_A:LON_B,LAT_B).
# On coarse grids the commented ones keep the Mediterranean, Red Sea,
# Black Sea and Hudson Bay connected to the open ocean; without them those
# seas are filled as land.
STRAITS=(
    "indonesian_throughflow:85,155,-15,15:90,-5:150,5"
    # "gibraltar:340,10,30,42:345,36:5,38"
    # "bab_el_mandeb:35,55,5,25:38,20:52,12"
    # "bosporus:20,42,35,47:35,43:25,37"
    # "hudson_strait:270,305,50,68:275,58:302,58"
)
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [-o OUTPUT_DIR] [--nproc N] [-r JCM_RESOLUTION]
                        [--rgll-nlat NLAT] [--rgll-nlon NLON]
                        [--rgll-lat-spacing equal|offset|gaussian] [--rgll-lon-centered]
                        [--rgll-rotation-deg D] [--rgll-axis-longitude-deg L]
                        [--land-threshold T] [--min-depth M]

  -o, --output-dir DIR         directory for grids, weights and outputs (default: ${OUTPUT_DIR})
      --nproc N                MPI ranks for ESMF_RegridWeightGen (default: ${NPROC})
  -r, --jcm-resolution N       atm grid: JCM spectral truncation (default: ${JCM_RESOLUTION})
      --rgll-nlat NLAT         surface grid latitude count (default: ${RGLL_NLAT})
      --rgll-nlon NLON         surface grid longitude count (default: ${RGLL_NLON})
      --rgll-lat-spacing S     surface grid latitude faces: equal, offset or gaussian (default: ${RGLL_LAT_SPACING})
      --rgll-lon-centered      center the first surface longitude cell on 0 deg
      --rgll-rotation-deg D    surface grid pole rotation, deg (default: ${RGLL_ROTATION_DEG})
      --rgll-axis-longitude-deg L  longitude of the rotation axis, deg (default: ${RGLL_ROTATION_AXIS_LONGITUDE_DEG})
      --land-threshold T       land if land_fraction >= T (default: ${LAND_THRESHOLD})
      --min-depth M            minimum ocean depth in m (default: ${MIN_DEPTH})
  -h, --help                   show this help message

Required straits are set in the STRAITS array at the top of this script.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output-dir)           OUTPUT_DIR="$2";                       shift 2 ;;
        --nproc)                   NPROC="$2";                            shift 2 ;;
        -r|--jcm-resolution)       JCM_RESOLUTION="$2";                   shift 2 ;;
        --rgll-nlat)               RGLL_NLAT="$2";                        shift 2 ;;
        --rgll-nlon)               RGLL_NLON="$2";                        shift 2 ;;
        --rgll-lat-spacing)        RGLL_LAT_SPACING="$2";                 shift 2 ;;
        --rgll-lon-centered)       RGLL_LON_CENTERED=1;                   shift ;;
        --rgll-rotation-deg)       RGLL_ROTATION_DEG="$2";                shift 2 ;;
        --rgll-axis-longitude-deg) RGLL_ROTATION_AXIS_LONGITUDE_DEG="$2"; shift 2 ;;
        --land-threshold)          LAND_THRESHOLD="$2";                   shift 2 ;;
        --min-depth)               MIN_DEPTH="$2";                        shift 2 ;;
        -h|--help)                 usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$RGLL_LAT_SPACING" in
    equal)    RGLL_LAT_BOUNDS_EXPR="equally_spaced_bounds(${RGLL_NLAT}, -90.0, 90.0)" ;;
    offset)   RGLL_LAT_BOUNDS_EXPR="offset_equally_spaced_bounds(${RGLL_NLAT}, -90.0, 90.0)" ;;
    gaussian) RGLL_LAT_BOUNDS_EXPR="gaussian_latitude_bounds(${RGLL_NLAT})" ;;
    *)
        echo "error: unknown --rgll-lat-spacing: ${RGLL_LAT_SPACING} (expected equal, offset or gaussian)" >&2
        exit 1
        ;;
esac
if [ "$RGLL_LON_CENTERED" -eq 1 ]; then
    RGLL_LON_LO="-180.0/${RGLL_NLON}"
    RGLL_LON_HI="360.0 - 180.0/${RGLL_NLON}"
else
    RGLL_LON_LO="0.0"
    RGLL_LON_HI="360.0"
fi

CLEANUP_ARGS=(--land-threshold "$LAND_THRESHOLD" --min-depth "$MIN_DEPTH")
for strait in "${STRAITS[@]}"; do
    CLEANUP_ARGS+=(--strait "$strait")
done

set -x

mkdir -p "$OUTPUT_DIR"

atm_scrip="${OUTPUT_DIR}/JCM_T${JCM_RESOLUTION}.SCRIP.nc"
sfc_scrip="${OUTPUT_DIR}/RotatedGaussianLatLon.SCRIP.nc"

echo "Generating atm grid: JCM T${JCM_RESOLUTION} ..."
python3 -c "
from EarthSystemGrids.JCMGrid import generate_JCMGrid
generate_JCMGrid(${JCM_RESOLUTION}).write_to_SCRIP_grid_file('${atm_scrip}', flatten=True)
"

echo "Generating surface grid: rotated Gaussian lat-lon ..."
python3 -c "
from EarthSystemGrids.GaussianLatLon import gaussian_latitude_bounds, equally_spaced_bounds, offset_equally_spaced_bounds
from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon
mesh = RotatedGaussianLatLon.generate_mesh(
    lat=${RGLL_LAT_BOUNDS_EXPR},
    lon=equally_spaced_bounds(${RGLL_NLON}, ${RGLL_LON_LO}, ${RGLL_LON_HI}),
    rotation_axis_longitude_deg=${RGLL_ROTATION_AXIS_LONGITUDE_DEG},
    rotation_deg=${RGLL_ROTATION_DEG},
)
mesh.write_to_SCRIP_grid_file('${sfc_scrip}', flatten=True)
"

"${SCRIPT_DIR}/run_part2.sh" \
    --atm-scrip "$atm_scrip" \
    --sfc-scrip "$sfc_scrip" \
    --output-dir "$OUTPUT_DIR" \
    --nproc "$NPROC" \
    -- "${CLEANUP_ARGS[@]}"
