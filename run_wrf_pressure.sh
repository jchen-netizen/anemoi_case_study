#!/bin/bash
# Usage: 
#   bash run_wrf_pressure.sh <base_path> <output_dir> <folder1> [folder2 ...]
# Example:
#   bash run_wrf_pressure.sh /Volumes/webdav/Share_Forecasts/WAC00WG-01 output.nc \
#       23051000 23051100 23061000 23081000

BASE_PATH="$1"
OUTPUT_DIR="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for FOLDER in "$@"; do

    echo ">>> ${FOLDER}"
    python3 "${SCRIPT_DIR}/wrf_pressure.py" "${BASE_PATH}/${FOLDER}00" "$OUTPUT_DIR"
done

echo "All done!"