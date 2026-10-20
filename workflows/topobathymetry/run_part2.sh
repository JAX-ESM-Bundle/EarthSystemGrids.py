#!/bin/bash
#
# Run Part 2 of the GEBCO preprocessing workflow end to end (see README.md):
#
#   02_generate_weights.sh    step 1: weights 0.1deg->sfc, sfc<->atm
#   03_regrid_to_surface.py   step 2: 0.1deg elevation/land_fraction -> sfc
#   04_clean_surface.py       steps 3-4: cleanup + final surface file
#   05_regrid_to_atm.py       step 5: sfc land_fraction/topography -> atm
#
# Run in an env with EarthSystemGrids importable, ESMF_RegridWeightGen and mpirun
# (e.g. the `jax` conda env). Arguments after `--` go to 04_clean_surface.py
# (e.g. -- --land-threshold 0.3 --min-depth 20).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --- inputs you may want to change ------------------------------------------
ATM_SCRIP=""
SFC_SCRIP=""
OUTPUT_DIR=part2_output
NPROC=16
INPUT_0P1=GEBCO_bathymetry_res-0p1.nc
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --atm-scrip FILE --sfc-scrip FILE [-o OUTPUT_DIR] [--nproc N]
                        [--input-0p1 FILE] [-- CLEANUP_ARGS...]

  --atm-scrip FILE     SCRIP file of the atmosphere grid (required)
  --sfc-scrip FILE     SCRIP file of the surface (ocn/lnd) grid (required)
  -o, --output-dir DIR directory for all outputs (default: ${OUTPUT_DIR})
  --nproc N            MPI ranks for ESMF_RegridWeightGen (default: ${NPROC})
  --input-0p1 FILE     Part 1 output (default: ${INPUT_0P1})
  -- CLEANUP_ARGS...   extra arguments passed to 04_clean_surface.py
  -h, --help           show this help message
EOF
}

CLEANUP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --atm-scrip)     ATM_SCRIP="$2";  shift 2 ;;
        --sfc-scrip)     SFC_SCRIP="$2";  shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --nproc)         NPROC="$2";      shift 2 ;;
        --input-0p1)     INPUT_0P1="$2";  shift 2 ;;
        --)              shift; CLEANUP_ARGS=("$@"); break ;;
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

atm_label=$(basename "$ATM_SCRIP" .SCRIP.nc)
sfc_label=$(basename "$SFC_SCRIP" .SCRIP.nc)

set -x

"${SCRIPT_DIR}/02_generate_weights.sh" \
    --atm-scrip "$ATM_SCRIP" --sfc-scrip "$SFC_SCRIP" \
    --output-dir "$OUTPUT_DIR" --nproc "$NPROC"

python3 "${SCRIPT_DIR}/03_regrid_to_surface.py" \
    --input "$INPUT_0P1" \
    --weight "${OUTPUT_DIR}/weight_algo-conserve_GEBCO_0p1_to_${sfc_label}.nc" \
    --sfc-scrip "$SFC_SCRIP" \
    --output "${OUTPUT_DIR}/sfc_regridded_${sfc_label}.nc"

python3 "${SCRIPT_DIR}/04_clean_surface.py" \
    --input "${OUTPUT_DIR}/sfc_regridded_${sfc_label}.nc" \
    --sfc-scrip "$SFC_SCRIP" \
    --area-weight "${OUTPUT_DIR}/weight_algo-conserve_GEBCO_0p1_to_${sfc_label}.nc" \
    --output "${OUTPUT_DIR}/sfc_topo_${sfc_label}.nc" \
    "${CLEANUP_ARGS[@]}"

python3 "${SCRIPT_DIR}/05_regrid_to_atm.py" \
    --input "${OUTPUT_DIR}/sfc_topo_${sfc_label}.nc" \
    --weight "${OUTPUT_DIR}/weight_algo-conserve_${sfc_label}_to_${atm_label}.nc" \
    --atm-scrip "$ATM_SCRIP" \
    --output "${OUTPUT_DIR}/atm_topo_${atm_label}.nc"

echo "Done. Surface and atm files written to ${OUTPUT_DIR}/"
