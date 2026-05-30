#!/usr/bin/env python3
"""
fill_obs.py — Fill observation values from CSV into NetCDF file.

Usage:  
    python3 fill_obs.py <input.csv> <input.nc> <output.nc>

Example: 
    python3 fill_obs.py observation.csv verif.nc verif_with_obs.nc

Behaviour:
    - reads obs from input.csv and fills them into input.nc, writing to output.nc
    - matches obs to NetCDF grid points by station ID and valid time
        init_time + leadtime hours == obs_time
        one CSV row can fill multiple (init_time, leadtime) pairs if they have the same valid time
    - obs CSV must have columns: obs_time, location, obs


CSV format (comma-delimited):
    obs_time, location, obs
    2023-05-08 06:00:00, 129, 12.3
"""

import sys
import shutil
import numpy as np
import pandas as pd
import netCDF4 as nc

csv_file   = sys.argv[1]
nc_in      = sys.argv[2]
nc_out     = sys.argv[3]

# Load obs CSV
obs_df = pd.read_csv(csv_file, parse_dates=["obs_time"])

# Copy input NetCDF to output NetCDF (so we can modify it in-place)
shutil.copy2(nc_in, nc_out)

# Fill obs
filled = 0

with nc.Dataset(nc_out, "r+") as ds:
    times     = ds["time"][:]        # init times, seconds since epoch 
    leadtimes = ds["leadtime"][:]    # hours since init             
    locations = ds["location"][:]    # station IDs                  
    obs_var   = ds["obs"]            # (time, leadtime, location)

    for _, row in obs_df.iterrows():
        obs_time = row["obs_time"]   
        loc_id   = int(row["location"])
        obs_val  = float(row["obs"])

        # Match location
        loc_matches = np.where(locations == loc_id)[0]
        if len(loc_matches) == 0:
            continue
        loc_idx = loc_matches[0]

        # Check every (init_time, leadtime) pair whose valid time == obs_time
        for t_idx, init_unix in enumerate(times):
            init_dt = pd.Timestamp(int(init_unix), unit="s", tz="UTC").tz_localize(None)
            for l_idx, lead_h in enumerate(leadtimes):
                valid_dt = init_dt + pd.Timedelta(hours=float(lead_h))
                if valid_dt == obs_time:
                    obs_var[t_idx, l_idx, loc_idx] = obs_val
                    filled += 1

print(f"Done → {nc_out}  ({filled} obs filled)") 