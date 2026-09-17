#!/bin/bash
#
# Part 2, step 1: generate the regridding weights the rest of Part 2 uses.
#
#   a. GEBCO 0.1deg -> surface grid   (conserve)
#   b. surface grid -> atm grid       (bilinear, conserve)
#   c. atm grid     -> surface grid   (bilinear, conserve)
#
# The atm and surface grids are taken as ready-made SCRIP files (e.g. from
# script/generate_fractional_mask_via_ncremap*.sh); only the 0.1deg source
# grid's SCRIP file is built here, from Part 1's node-centered convention
# with half-width polar caps as the real pole-row bounds. Weight files are
# named like the other scripts in this repo:
# weight_algo-${method}_${src}_to_${dst}.nc
#
# The 0.1deg grid has ~6.5M cells, so ESMF_RegridWeightGen runs under
# mpirun when --nproc > 1. --ignore_degenerate lets ESMF accept the pole-cap
# cells, whose two polar corners coincide.
#
# Needs ESMF_RegridWeightGen (and mpirun for --nproc > 1) on PATH, e.g. the
# `jax` conda env.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --- inputs you may want to change ------------------------------------------
ATM_SCRIP=""
SFC_SCRIP=""
ATM_LABEL=""                       # default: ATM_SCRIP basename without .SCRIP.nc
SFC_LABEL=""                       # default: SFC_SCRIP basename without .SCRIP.nc
OUTPUT_DIR=part2_output
NPROC=16

SRC_LABEL=GEBCO_0p1
WEIGHT_METHODS=(bilinear conserve) # sfc <-> atm methods
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --atm-scrip FILE --sfc-scrip FILE [--atm-label L] [--sfc-label L]
                        [-o OUTPUT_DIR] [--nproc N]

  --atm-scrip FILE     SCRIP file of the atmosphere grid (required)
  --sfc-scrip FILE     SCRIP file of the surface (ocn/lnd) grid (required)
  --atm-label L        name used in weight file names (default: SCRIP basename without .SCRIP.nc)
  --sfc-label L        name used in weight file names (default: SCRIP basename without .SCRIP.nc)
  -o, --output-dir DIR directory for the 0.1deg SCRIP file and weight files (default: ${OUTPUT_DIR})
  --nproc N            MPI ranks for ESMF_RegridWeightGen; 1 runs without mpirun (default: ${NPROC})
  -h, --help           show this help message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --atm-scrip)     ATM_SCRIP="$2";  shift 2 ;;
        --sfc-scrip)     SFC_SCRIP="$2";  shift 2 ;;
        --atm-label)     ATM_LABEL="$2";  shift 2 ;;
        --sfc-label)     SFC_LABEL="$2";  shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --nproc)         NPROC="$2";      shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$ATM_SCRIP" ] || [ -z "$SFC_SCRIP" ]; then
    echo "error: --atm-scrip and --sfc-scrip are required" >&2
    usage >&2
    exit 1
fi
for f in "$ATM_SCRIP" "$SFC_SCRIP"; do
    if [ ! -f "$f" ]; then
        echo "error: $f not found" >&2
        exit 1
    fi
done

ATM_LABEL=${ATM_LABEL:-$(basename "$ATM_SCRIP" .SCRIP.nc)}
SFC_LABEL=${SFC_LABEL:-$(basename "$SFC_SCRIP" .SCRIP.nc)}

if [ "$NPROC" -gt 1 ]; then
    ESMF=(mpirun -np "$NPROC" ESMF_RegridWeightGen)
else
    ESMF=(ESMF_RegridWeightGen)
fi

set -x

mkdir -p "$OUTPUT_DIR"

src_scrip="${OUTPUT_DIR}/${SRC_LABEL}.SCRIP.nc"

echo "Generating GEBCO 0.1deg SCRIP grid ..."
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 -c "
from common import build_0p1deg_mesh
build_0p1deg_mesh().write_to_SCRIP_grid_file('${src_scrip}', flatten=True)
"

echo "Generating ${SRC_LABEL} -> ${SFC_LABEL} conservative weights ..."
"${ESMF[@]}" --ignore_degenerate \
    -s "${src_scrip}" -d "${SFC_SCRIP}" -m conserve \
    -w "${OUTPUT_DIR}/weight_algo-conserve_${SRC_LABEL}_to_${SFC_LABEL}.nc"

echo "Generating ${SFC_LABEL} <-> ${ATM_LABEL} weights ..."
for method in "${WEIGHT_METHODS[@]}"; do
    "${ESMF[@]}" --ignore_degenerate \
        -s "${SFC_SCRIP}" -d "${ATM_SCRIP}" -m "${method}" \
        -w "${OUTPUT_DIR}/weight_algo-${method}_${SFC_LABEL}_to_${ATM_LABEL}.nc"
    "${ESMF[@]}" --ignore_degenerate \
        -s "${ATM_SCRIP}" -d "${SFC_SCRIP}" -m "${method}" \
        -w "${OUTPUT_DIR}/weight_algo-${method}_${ATM_LABEL}_to_${SFC_LABEL}.nc"
done

echo "Done. Weights written to ${OUTPUT_DIR}/"
