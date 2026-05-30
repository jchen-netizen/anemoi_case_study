#!/usr/bin/env python3
"""
extract_obs.py — Extract BC Wildfire weather stations data in CSV format.

Usage:
    python3 extract_obs.py <data_files> <variable> <output.csv>

Arguments:
    data_files      One or more data files containing weather station data (e.g. 2023-05-*.csv)
    variable        Variable name to extract                       (e.g. temperature)
    output.csv      Output CSV file                                (e.g. output.csv)

Examples:
    python3 extract_obs.py 2023-05-*.csv temperature output.csv

Behaviour:
    - renames columns to lowercase and replaces spaces with underscores
    - filters to only include desired stations (see STATIONS list below)
    - extracts only the specified variable (e.g. temperature), 
        date_time (e.g. 2023-05-08 00:00:00) and station_code (e.g. 129)
    - appends all data to output.csv
        output.csv will have columns: date_time, station_code, obs

Stations (hardcoded — add more to STATIONS list):
    id=129  Pink Mountain  lat=56.94  lon=-122.70  alt=960.10 m
    id=120  Silver        lat=57.37  lon=-121.41  alt=835.00 m
    id=131  Muskwa        lat=57.88  lon=-123.62  alt=769.00 m

Variables:
    station_code,
    date_time,
    precipitation,
    temperature,
    relative_humidity,
    wind_speed,
    wind_direction,
    fine_fuel_moisture_code,
    initial_spread_index,
    fire_weather_index,
    duff_moisture_code,
    drought_code,
    buildup_index,
    danger_rating,
    rn_1_pluvio1,
    snow_depth,
    snow_depth_quality,
    precip_pluvio1_status,
    precip_pluvio1_total,
    rn_1_pluvio2,
    precip_pluvio2_status,
    precip_pluvio2_total,
    rn_1_rit,
    precip_rit_status,
    precip_rit_total,
    precip_rgt,
    solar_radiation_licor,
    solar_radiation_cm3
"""

import sys
import os
import pandas as pd

STATION_IDS = [129, 120, 131]

data_files = sys.argv[1:-2]
variable = sys.argv[-2]
output_csv = sys.argv[-1]

# Read and concatenate all input files
frames = []
for fpath in sorted(data_files):
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df[df['station_code'].isin(STATION_IDS)]
    df = df[['date_time', 'station_code', variable]]
    
    # change variable name to obs for easier concatenation later
    df = df.rename(columns={variable: 'obs', 
                            'station_code': 'location', 
                            'date_time': 'obs_time'})
    
    # convert to UTC (data is in local time, which is UTC-7)
    df["obs_time"] = pd.to_datetime(df["obs_time"], format="%Y%m%d%H") + pd.Timedelta(hours=7) 
    
    frames.append(df[['obs_time', 'location', 'obs']]) 

new_data = pd.concat(frames, ignore_index=True)

# Probably not needed, but just in case, append to existing output file, drop duplicates
if os.path.exists(output_csv):
    existing_data = pd.read_csv(output_csv, parse_dates=["obs_time"])
    new_data = pd.concat([existing_data, new_data], ignore_index=True)

new_data = (new_data
    .drop_duplicates(subset=["obs_time", "location"])
    .sort_values(["location", "obs_time"])
    .reset_index(drop=True))
 
new_data.to_csv(output_csv, index=False)
print(f"Done → {output_csv}  ({len(new_data)} rows)")