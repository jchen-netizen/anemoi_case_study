#!/usr/bin/env python3
"""
wrf2verif_wind.py — Extract WRF U/V wind into Verif-format NetCDF
                     (wind speed + wind direction, two output files).

Usage:
    python3 wrf2verif_wind.py <wrf_files> <init_time> <output.nc>

Arguments:
    wrf_files       One or more WRF output files (glob-expanded by shell, e.g. wrfout_d02_*)
    init_time       Forecast initialization time as YYYYMMDDHH  (e.g. 2023050800)
    output.nc       Output Verif-format NetCDF filename (base name).
                    Two files are actually written:
                        <output>_wspd.nc   -> fcst = wind speed (m/s)
                        <output>_wdir.nc   -> fcst = wind direction (deg, 0-360, from-north)

Examples:
    python3 wrf2verif_wind.py wrfout_d02_* 2023050800 output.nc
    python3 wrf2verif_wind.py wrfout_d02_2023-05-08_* 2023050800 output.nc

Behaviour:
    - If an output file already exists AND contains data, the new init time is
      MERGED in (appended along the time dimension) — same as original script.
    - Each WRF file contributes ONE valid time -> ONE leadtime step.
    - Leadtime is derived from the filename timestamp minus init_time (hours).
    - U (west_east_stag) and V (south_north_stag) are destaggered (averaged
      with their neighbour) onto the mass grid before use.
    - Wind is taken at model level LEVEL (0 = lowest model level, nearest
      surface). Change the LEVEL constant below if a different level is needed.
    - Observations are left as NaN (fill in later).

Stations (hardcoded — add more to STATIONS list):
    id=129  Pink Mountain  lat=56.94  lon=-122.70  alt=960.10 m
    id=120  Silver        lat=57.37  lon=-121.41  alt=835.00 m
    id=131  Muskwa        lat=57.88  lon=-123.62  alt=769.00 m
"""

import sys
import os
import re
import argparse
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timezone

# =============================================================
# STATIONS  — edit / extend this list as needed
# =============================================================
STATIONS = [
    {"id": 129, "name": "Pink Mountain", "lat": 56.94, "lon": -122.70, "alt": 960.10},
    {"id": 120, "name": "Silver",     "lat": 57.37, "lon": -121.41, "alt": 835.00},
    {"id": 131, "name": "Muskwa",     "lat": 57.88, "lon": -123.62, "alt": 769.00},
]

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Model (bottom_top) level to extract wind from. 0 = lowest model level
# (nearest the surface). Change this if you want a different level.
LEVEL = 0


# =============================================================
# HELPERS
# =============================================================

def parse_init_time(s):
    """Parse YYYYMMDDHH → datetime (UTC)."""
    return datetime.strptime(s, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def parse_valid_time_from_filename(filepath):
    """
    Extract valid time from WRF filename.
    Supports patterns like:
        wrfout_d02_2023-05-08_06:00:00
        wrfout_d02_2023-05-08_06_00_00
        wrfout_d02_2023-05-08_060000
    Returns a datetime (UTC).
    """
    basename = os.path.basename(filepath)

    # Try pattern: YYYY-MM-DD_HH:MM:SS  or  YYYY-MM-DD_HH_MM_SS
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})[:_](\d{2})[:_](\d{2})', basename)
    if m:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)

    # Try pattern: YYYY-MM-DD_HHMMSS
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{6})', basename)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = m.group(4)
        h, mi, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)

    raise ValueError(f"Cannot parse valid time from filename: {basename}")


def find_nearest_idx(ds, target_lat, target_lon):
    """Return (sn_idx, we_idx) of nearest WRF *mass* grid point.

    U and V are destaggered back onto this same mass grid (see
    extract_wind_from_wrf_files), so the indices found here apply to
    XLAT/XLONG as well as to the destaggered U/V fields.
    """
    xlat  = ds["XLAT"].values   # shape: (Time, south_north, west_east)
    xlong = ds["XLONG"].values

    # Use first time slice for the lat/lon grid
    dist = np.sqrt((xlat[0] - target_lat)**2 + (xlong[0] - target_lon)**2)
    min_idx = np.unravel_index(dist.argmin(), dist.shape)
    sn_idx, we_idx = int(min_idx[0]), int(min_idx[1])
    return sn_idx, we_idx


def seconds_since_epoch(dt):
    return int((dt - EPOCH).total_seconds())


def uv_to_speed_dir(u, v):
    """Convert eastward/northward wind components to speed (m/s) and
    meteorological direction (deg, 0-360, direction wind is FROM,
    0=N, 90=E, 180=S, 270=W)."""
    speed = np.sqrt(u * u + v * v)
    wdir = np.degrees(np.arctan2(-u, -v))
    wdir = np.mod(wdir, 360.0)
    return speed, wdir


# =============================================================
# CORE: read WRF files → (leadtimes, speed/dir per station)
# =============================================================

def extract_wind_from_wrf_files(wrf_files, init_time):
    """
    Returns:
        leadtimes  : sorted list of int lead hours
        spd_array  : np.ndarray shape (n_leads, n_stations), float32, m/s
        dir_array  : np.ndarray shape (n_leads, n_stations), float32, degrees
    """
    n_stations = len(STATIONS)

    # Pre-find nearest MASS grid indices using the first file (unchanged
    # from original script — U/V are destaggered onto this same grid below).
    print(f"  Finding nearest grid points using {wrf_files[0]} ...")
    ds0 = xr.open_dataset(wrf_files[0])
    station_indices = []
    for st in STATIONS:
        sn, we = find_nearest_idx(ds0, st["lat"], st["lon"])
        nearest_lat = float(ds0["XLAT"].values[0, sn, we])
        nearest_lon = float(ds0["XLONG"].values[0, sn, we])
        print(f"    {st['name']:20s}  target=({st['lat']:.4f},{st['lon']:.4f})"
              f"  nearest=({nearest_lat:.4f},{nearest_lon:.4f})  grid=({sn},{we})")
        station_indices.append((sn, we))
    ds0.close()

    # Read each file → one leadtime step
    spd_records = {}  # lead_hour (int) → np.array shape (n_stations,)
    dir_records = {}

    for fpath in sorted(wrf_files):
        try:
            valid_time = parse_valid_time_from_filename(fpath)
        except ValueError as e:
            print(f"  WARNING: {e} — skipping file.")
            continue

        lead_h = (valid_time - init_time).total_seconds() / 3600.0
        if lead_h < 0:
            print(f"  Skipping {os.path.basename(fpath)} (lead={lead_h:.1f}h < 0)")
            continue

        lead_int = int(round(lead_h))
        print(f"  {os.path.basename(fpath)}  valid={valid_time.strftime('%Y-%m-%d %H:%M')}  lead={lead_int}h")

        ds = xr.open_dataset(fpath)
        spd_vals = np.full(n_stations, np.nan, dtype=np.float32)
        dir_vals = np.full(n_stations, np.nan, dtype=np.float32)

        for i, (sn, we) in enumerate(station_indices):
            # Destagger U: average west_east_stag[we] and [we+1] -> mass point we
            u1 = float(ds["U"].isel(bottom_top=LEVEL, south_north=sn,
                                     west_east_stag=we).values.flat[0])
            u2 = float(ds["U"].isel(bottom_top=LEVEL, south_north=sn,
                                     west_east_stag=we + 1).values.flat[0])
            u = 0.5 * (u1 + u2)

            # Destagger V: average south_north_stag[sn] and [sn+1] -> mass point sn
            v1 = float(ds["V"].isel(bottom_top=LEVEL, south_north_stag=sn,
                                     west_east=we).values.flat[0])
            v2 = float(ds["V"].isel(bottom_top=LEVEL, south_north_stag=sn + 1,
                                     west_east=we).values.flat[0])
            v = 0.5 * (v1 + v2)

            speed, wdir = uv_to_speed_dir(u, v)
            spd_vals[i] = speed
            dir_vals[i] = wdir

        spd_records[lead_int] = spd_vals
        dir_records[lead_int] = dir_vals
        ds.close()

    if not spd_records:
        raise RuntimeError("No valid WRF files could be processed.")

    sorted_leads = sorted(spd_records.keys())
    spd_array = np.stack([spd_records[l] for l in sorted_leads], axis=0)  # (n_leads, n_stations)
    dir_array = np.stack([dir_records[l] for l in sorted_leads], axis=0)
    return sorted_leads, spd_array, dir_array


# =============================================================
# WRITE / MERGE VERIF NETCDF  (unchanged logic, generalised attrs)
# =============================================================

def write_verif_nc(output_file, init_time, variable, leadtimes, fcst_array):
    """
    Create or merge a VERIF-format NetCDF file.

    fcst_array shape: (n_leads, n_stations)
    leadtimes: sorted list of ints (hours)
    variable: "WSPD" or "WDIR" — controls units/long_name only.
    """
    n_locs   = len(STATIONS)
    ids      = np.array([s["id"]  for s in STATIONS], dtype=np.int32)
    lats     = np.array([s["lat"] for s in STATIONS], dtype=np.float32)
    lons     = np.array([s["lon"] for s in STATIONS], dtype=np.float32)
    alts     = np.array([s["alt"] for s in STATIONS], dtype=np.float32)

    init_unix = seconds_since_epoch(init_time)

    # === Determine global leadtime axis (union of existing + new) ===
    new_lead_arr = np.array(leadtimes, dtype=np.float32)

    if os.path.exists(output_file):
        # === MERGE mode =============================================================
        print(f"  Output file exists — merging into {output_file} ...")

        with nc.Dataset(output_file, "r") as existing:
            ex_times  = existing["time"][:]            # existing init times (unix sec)
            ex_leads  = existing["leadtime"][:]        # existing leadtime axis
            ex_fcst   = existing["fcst"][:]            # (time, leadtime, location)

        # Check this init time isn't already in the file
        if init_unix in ex_times:
            print(f"  WARNING: init time {init_time} already in {output_file}. Overwriting that slot.")
            t_idx = int(np.where(ex_times == init_unix)[0][0])
            overwrite_slot = t_idx
        else:
            overwrite_slot = None

        # Build merged leadtime axis
        merged_leads = np.union1d(ex_leads, new_lead_arr).astype(np.float32)
        n_leads_new  = len(merged_leads)
        n_times_ex   = len(ex_times)

        # Rebuild fcst array on merged leadtime axis
        if overwrite_slot is not None:
            merged_times = ex_times.copy()
            n_times_out  = n_times_ex
        else:
            merged_times = np.append(ex_times, init_unix).astype(np.int32)
            n_times_out  = n_times_ex + 1

        merged_fcst = np.full((n_times_out, n_leads_new, n_locs), np.nan, dtype=np.float32)

        # Re-map existing data onto merged leadtime axis
        for old_li, old_lead in enumerate(ex_leads):
            new_li = int(np.where(merged_leads == old_lead)[0][0])
            for ti in range(n_times_ex):
                if overwrite_slot is not None and ti == overwrite_slot:
                    continue
                merged_fcst[ti, new_li, :] = ex_fcst[ti, old_li, :]

        # Insert new data
        new_t_idx = overwrite_slot if overwrite_slot is not None else n_times_ex
        for new_li_idx, lead_val in enumerate(new_lead_arr):
            ml_idx = int(np.where(merged_leads == lead_val)[0][0])
            merged_fcst[new_t_idx, ml_idx, :] = fcst_array[new_li_idx, :]

        # Re-write file
        os.remove(output_file)
        _write_nc_file(output_file, variable, merged_times, merged_leads,
                       ids, lats, lons, alts, merged_fcst)

    else:
        # === CREATE mode =============================================================
        print(f"  Creating new output file: {output_file}")
        times_arr = np.array([init_unix], dtype=np.int32)
        # fcst shape must be (1, n_leads, n_locs)
        fcst_3d   = fcst_array[np.newaxis, :, :]  # (1, n_leads, n_locs)
        _write_nc_file(output_file, variable, times_arr, new_lead_arr,
                       ids, lats, lons, alts, fcst_3d)


def _write_nc_file(output_file, variable, times_arr, leads_arr,
                   ids, lats, lons, alts, fcst_3d):
    """Low-level writer — always creates a fresh file."""
    n_times, n_leads, n_locs = fcst_3d.shape

    if variable == "WSPD":
        units, long_name, std_name = "m s-1", "Wind Speed", "wind_speed"
    elif variable == "WDIR":
        units, long_name, std_name = "degrees", "Wind Direction", "wind_from_direction"
    else:
        units, long_name, std_name = "unknown", variable, variable

    with nc.Dataset(output_file, "w", format="NETCDF4") as out:

        # Dimensions
        out.createDimension("time",     None)      # UNLIMITED
        out.createDimension("leadtime", n_leads)
        out.createDimension("location", n_locs)

        # time
        v = out.createVariable("time", "i4", ("time",))
        v[:] = times_arr
        v.units    = "seconds since 1970-01-01 00:00:00 +00:00"
        v.long_name = "Forecast initialization time"

        # leadtime
        v = out.createVariable("leadtime", "f4", ("leadtime",))
        v[:] = leads_arr
        v.units    = "hours"
        v.long_name = "Hours since forecast initialization"

        # location
        v = out.createVariable("location", "i4", ("location",))
        v[:] = ids
        v.long_name = "Station ID"

        # lat / lon / altitude
        v = out.createVariable("lat", "f4", ("location",))
        v[:] = lats
        v.units = "degrees_north"

        v = out.createVariable("lon", "f4", ("location",))
        v[:] = lons
        v.units = "degrees_east"

        v = out.createVariable("altitude", "f4", ("location",))
        v[:] = alts
        v.units = "meters"

        # obs — intentionally left as NaN
        v = out.createVariable("obs", "f4", ("time", "leadtime", "location"),
                               fill_value=np.nan)
        v[:] = np.full((n_times, n_leads, n_locs), np.nan, dtype=np.float32)
        v.long_name = "Observations (to be filled)"
        v.units     = units

        # fcst
        v = out.createVariable("fcst", "f4", ("time", "leadtime", "location"),
                               fill_value=np.nan)
        v[:] = fcst_3d
        v.long_name = f"WRF {long_name} forecast"
        v.units     = units
        v.wrf_variable = variable

        # Global attributes (Verif standard)
        out.long_name     = long_name
        out.standard_name = std_name
        out.units         = units
        out.verif_version = "1.0.0"
        out.source        = "WRF model output"
        out.created_by    = "wrf2verif_wind.py"

    print(f"  Done → {output_file}  "
          f"(times={n_times}, leadtimes={n_leads}, locations={n_locs})")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert WRF U/V output to Verif-format wind speed & direction NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("netcdf_files",   nargs="+", help="WRF output file(s)")
    parser.add_argument("initialized_time", help="Init time as YYYYMMDDHH (e.g. 2023050800)")
    parser.add_argument("output_netcdf_file", help="Output base filename, e.g. output.nc "
                                                     "(writes output_wspd.nc and output_wdir.nc)")

    args = parser.parse_args()

    wrf_files   = args.netcdf_files
    init_time   = parse_init_time(args.initialized_time)
    output_base = args.output_netcdf_file

    root, ext = os.path.splitext(output_base)
    if not ext:
        ext = ".nc"
    spd_file = f"{root}_wspd{ext}"
    dir_file = f"{root}_wdir{ext}"

    print(f"\n{'='*60}")
    print(f"  WRF → Verif wind converter")
    print(f"  Init time   : {init_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Level index : {LEVEL} (bottom_top)")
    print(f"  Files       : {len(wrf_files)} file(s)")
    print(f"  Output      : {spd_file}, {dir_file}")
    print(f"{'='*60}\n")

    print("Step 1: Extracting U/V from WRF files ...")
    leadtimes, spd_array, dir_array = extract_wind_from_wrf_files(wrf_files, init_time)
    print(f"  → Lead hours: {leadtimes}")

    print("\nStep 2: Writing Verif-format NetCDF (wind speed) ...")
    write_verif_nc(spd_file, init_time, "WSPD", leadtimes, spd_array)

    print("\nStep 3: Writing Verif-format NetCDF (wind direction) ...")
    write_verif_nc(dir_file, init_time, "WDIR", leadtimes, dir_array)

    print("\nDone!\n")


if __name__ == "__main__":
    main()