#!/bin/bash
# Usage:
#   bash run_anemoi2verif_wind.sh <base_path> <output.nc> <start_date> <end_date> [extra anemoi2verif_wind.py args]
#
# <start_date> / <end_date> use the same YYMMDDHH format as
# run_wrf2verif_wind.sh (2-digit year, month, day, hour). The HH from
# <start_date> is kept constant and applied to every day in the range.
#
# All extra arguments after <end_date> are passed straight through to
# anemoi2verif_wind.py, e.g. --level 850.
#
# Example:
#   bash run_anemoi2verif_wind.sh /Volumes/webdav/Share_Forecasts/anemoi_runs \
#       output.nc 23051000 23081000
#   bash run_anemoi2verif_wind.sh /Volumes/webdav/Share_Forecasts/anemoi_runs \
#       output.nc 23051000 23081000 --level 850

BASE_PATH="$1"
OUTPUT_NC="$2"
START="$3"
END="$4"
shift 4 2>/dev/null
EXTRA_ARGS=("$@")

if [ -z "$BASE_PATH" ] || [ -z "$OUTPUT_NC" ] || [ -z "$START" ] || [ -z "$END" ]; then
    echo "Usage: bash run_anemoi2verif_wind.sh <base_path> <output.nc> <start_date YYMMDDHH> <end_date YYMMDDHH> [extra args]"
    exit 1
fi

if [ "${#START}" -ne 8 ] || [ "${#END}" -ne 8 ]; then
    echo "ERROR: start_date/end_date must be 8 digits, YYMMDDHH (e.g. 23051000)."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Convert YYMMDDHH -> YYYYMMDDHH for 6-hour stepping (century assumed 20YY)
start_ymdh="20${START}"
end_ymdh="20${END}"

# Portable 6-hour increment: prefer GNU date, fall back to BSD date (macOS)
if date -d "$start_ymdh" >/dev/null 2>&1; then
    DATE_MODE="gnu"
else
    DATE_MODE="bsd"
fi

if [ "$start_ymdh" -gt "$end_ymdh" ] 2>/dev/null; then
    echo "ERROR: start_date is after end_date."
    exit 1
fi

current="$start_ymdh"

while :; do
    FILE="${BASE_PATH}/${current:0:8}T${current:8:2}.nc"

    if [ -e "$FILE" ]; then
        echo ">>> ${current:0:8}T${current:8:2}.nc"
        python3 "${SCRIPT_DIR}/anemoi2verif_wind.py" "$FILE" "$OUTPUT_NC" "${EXTRA_ARGS[@]}"
    else
        echo ">>> ${current:0:8}T${current:8:2}.nc  -- not found, skipping"
    fi

    [ "$current" = "$end_ymdh" ] && break

    if [ "$DATE_MODE" = "gnu" ]; then
        current=$(date -d "${current} +6 hours" +%Y%m%d%H)
    else
        current=$(date -j -v+6H -f "%Y%m%d%H" "$current" +%Y%m%d%H)
    fi
done

echo "All done!"