#!/bin/bash
# Usage:
#   bash run_wrf2verif.sh <base_path> <variable> <output.nc> <start_date> <end_date>
#
# <start_date> / <end_date> use the same folder-name format as the WRF run
# folders, e.g. 23051000  (YYMMDDHH — 2-digit year, month, day, hour).
# The HH from <start_date> is kept constant and applied to every day in
# the range. All daily folders from <start_date> to <end_date> are
# processed, INCLUSIVE. If <start_date> == <end_date>, only that one
# folder is processed.
#
# Example:
#   bash run_wrf2verif.sh /Volumes/webdav/Share_Forecasts/WAC00WG-01 T2 output.nc \
#       23051000 23081000

BASE_PATH="$1"
VARIABLE="$2"
OUTPUT_NC="$3"
START="$4"
END="$5"

if [ -z "$BASE_PATH" ] || [ -z "$VARIABLE" ] || [ -z "$OUTPUT_NC" ] || [ -z "$START" ] || [ -z "$END" ]; then
    echo "Usage: bash run_wrf2verif.sh <base_path> <variable> <output.nc> <start_date YYMMDDHH> <end_date YYMMDDHH>"
    exit 1
fi

if [ "${#START}" -ne 8 ] || [ "${#END}" -ne 8 ]; then
    echo "ERROR: start_date/end_date must be 8 digits, YYMMDDHH (e.g. 23051000)."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Hour (HH) portion is taken from START and kept constant for every day.
HH="${START:6:2}"

# Convert YYMMDDHH -> YYYYMMDD for date arithmetic (century assumed 20YY)
start_ymd="20${START:0:6}"
end_ymd="20${END:0:6}"

# Portable day-increment: prefer GNU date, fall back to BSD date (macOS)
if date -d "$start_ymd" >/dev/null 2>&1; then
    DATE_MODE="gnu"
else
    DATE_MODE="bsd"
fi

if [ "$start_ymd" -gt "$end_ymd" ] 2>/dev/null; then
    echo "ERROR: start_date is after end_date."
    exit 1
fi

current="$start_ymd"

while :; do
    FOLDER="${current:2}${HH}"    # strip century -> YYMMDDHH
    INIT_TIME="20${FOLDER}"
    FILES=("${BASE_PATH}/${FOLDER}"/wrfout_d02_*00)

    if [ -e "${FILES[0]}" ]; then
        echo ">>> ${FOLDER}  (init: ${INIT_TIME}, ${#FILES[@]} files)"
        python3 "${SCRIPT_DIR}/wrf2verif.py" "${FILES[@]}" "$INIT_TIME" "$VARIABLE" "$OUTPUT_NC"
    else
        echo ">>> ${FOLDER}  (init: ${INIT_TIME})  -- no files found, skipping"
    fi

    [ "$current" = "$end_ymd" ] && break

    if [ "$DATE_MODE" = "gnu" ]; then
        current=$(date -d "${current} +1 day" +%Y%m%d)
    else
        current=$(date -j -v+1d -f "%Y%m%d" "$current" +%Y%m%d)
    fi
done

echo "All done!"