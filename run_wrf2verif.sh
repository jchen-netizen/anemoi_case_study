#!/bin/bash
# Usage: bash run_wrf2verif.sh <base_path> <variable> <output.nc> <folder1> [folder2 ...]
# Example:
#   bash run_wrf2verif.sh /Volumes/webdav/Share_Forecasts/WAC00WG-01 T2 output.nc \
#       23051000 23051100 23061000 23081000

BASE_PATH="$1"
VARIABLE="$2"
OUTPUT_NC="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for FOLDER in "$@"; do
    INIT_TIME="20${FOLDER}"
    FILES=("${BASE_PATH}/${FOLDER}"/wrfout_d02_*00)

    echo ">>> ${FOLDER}  (init: ${INIT_TIME}, ${#FILES[@]} files)"
    python3 "${SCRIPT_DIR}/wrf2verif.py" "${FILES[@]}" "$INIT_TIME" "$VARIABLE" "$OUTPUT_NC"
done