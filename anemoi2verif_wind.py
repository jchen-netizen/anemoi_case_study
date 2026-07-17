#!/usr/bin/env python3
"""
anemoi2verif_wind.py — Extract anemoi/ECMWF-style U/V wind into Verif-format
                        NetCDF (wind speed + wind direction, two output files).

Adapted from wrf2verif_wind.py the same way anemoi2verif.py was adapted from
wrf2verif.py

Usage:
    python3 anemoi2verif_wind.py <nc_files> <output.nc> [--level LEVEL]

Arguments:
    nc_files        One or more source files e.g. 20220701T00.nc 20220702T00.nc
    output.nc       Output Verif-format NetCDF base filename.
                     Two files are actually written:
                         <output>_wspd.nc   -> fcst = wind speed (m/s)
                         <output>_wdir.nc   -> fcst = wind direction (deg, 0-360, from-north)

Options:
    --level LEVEL     Which u/v pair to use. One of: 10m (default, surface
                       10u/10v), 50, 100, 250, 500, 850 (pressure-level
                       u_###/v_### in hPa).

Examples:
    python3 anemoi2verif_wind.py 20220701T00.nc output.nc
    python3 anemoi2verif_wind.py 20220701T00.nc 20220702T00.nc output.nc --level 850

Behaviour:
    - If an output file already exists AND contains data, the new init time
      is MERGED in (appended along the time dimension).
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
import netCDF4 as nc
from datetime import datetime

# =============================================================
# STATIONS  — edit / extend this list as needed
# =============================================================
STATIONS = [
    {"id": 129, "name": "Pink Mountain", "lat": 56.94, "lon": -122.70, "alt": 960.10},
    {"id": 120, "name": "Silver",     "lat": 57.37, "lon": -121.41, "alt": 835.00},
    {"id": 131, "name": "Muskwa",     "lat": 57.88, "lon": -123.62, "alt": 769.00},
]

# All datetimes in this script are naive and implicitly UTC (matches
# fill_obs.py's convention of tz_localize(None) on everything).
EPOCH = datetime(1970, 1, 1)

LEVEL_CHOICES = ["10m", "50", "100", "250", "500", "850"]


# =============================================================
# HELPERS
# =============================================================

def uv_variable_names(level):
    """Map a --level choice to the anemoi u/v variable names."""
    if level == "10m":
        return "10u", "10v"
    return f"u_{level}", f"v_{level}"


def parse_init_time(s):
    """Parse YYYYMMDDHH -> naive datetime (UTC implied)."""
    return datetime.strptime(s, "%Y%m%d%H")


def parse_init_time_from_filename(filepath):
    """
    Extract init time from a filename like 20220701T00.nc or 20230508T00.nc
    (YYYYMMDDTHH).
    """
    basename = os.path.basename(filepath)
    m = re.search(r'(\d{4})(\d{2})(\d{2})T(\d{2})', basename)
    if m:
        y, mo, d, h = (int(x) for x in m.groups())
        return datetime(y, mo, d, h)
    raise ValueError(
        f"Cannot parse init time from filename: {basename}. "
        f"Expected a pattern like YYYYMMDDTHH (e.g. 20220701T00.nc). "
        f"Use --init-time YYYYMMDDHH to specify it explicitly instead."
    )


def find_nearest_station_indices(lat_arr, lon_arr, stations):
    """Return a list of flat indices into lat_arr/lon_arr, one per station.
    Simple degree-space nearest neighbour."""
    indices = []
    for st in stations:
        dist = np.sqrt((lat_arr - st["lat"]) ** 2 + (lon_arr - st["lon"]) ** 2)
        idx = int(np.argmin(dist))
        nearest_lat, nearest_lon = float(lat_arr[idx]), float(lon_arr[idx])
        print(f"    {st['name']:20s}  target=({st['lat']:.4f},{st['lon']:.4f})"
              f"  nearest=({nearest_lat:.4f},{nearest_lon:.4f})  index={idx}")
        indices.append(idx)
    return indices


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


STATION_DIM_NAMES = ("values", "station", "location", "point", "cell")


def _to_plain_datetime(t):
    t = np.atleast_1d(t)[0]
    return datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)


def decode_cf_times(time_var):
    """Decode a CF-style time variable (absolute times) into plain datetimes."""
    calendar = getattr(time_var, "calendar", "standard")
    decoded = nc.num2date(time_var[:], units=time_var.units, calendar=calendar,
                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    decoded = np.atleast_1d(decoded)
    return np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in decoded])


def get_leadtimes_and_init_time(ds, filename_init_time):
    """
    Returns (leadtime_hours, time_dim_name, resolved_init_time).
    Supports 'time' (absolute) OR 'lead_time' + 'initial_date' (anemoi-style).
    """
    if "time" in ds.variables:
        valid_times = decode_cf_times(ds.variables["time"])
        leadtime_hours = np.array(
            [(vt - filename_init_time).total_seconds() / 3600.0 for vt in valid_times]
        )
        return leadtime_hours, "time", filename_init_time

    if "lead_time" in ds.variables:
        lt_var = ds.variables["lead_time"]
        leadtime_hours = lt_var[:].astype(np.float64)
        units = getattr(lt_var, "units", "hours").lower()
        if "second" in units:
            leadtime_hours = leadtime_hours / 3600.0
        elif "minute" in units:
            leadtime_hours = leadtime_hours / 60.0

        init_time = filename_init_time
        if "initial_date" in ds.variables:
            id_var = ds.variables["initial_date"]
            try:
                file_init_time = _to_plain_datetime(decode_cf_times(id_var))
                if filename_init_time is not None and file_init_time != filename_init_time:
                    print(f"  NOTE: file's own initial_date ({file_init_time}) differs from "
                          f"the filename/--init-time value ({filename_init_time}); "
                          f"using the file's initial_date.", file=sys.stderr)
                init_time = file_init_time
            except Exception as e:
                print(f"  WARNING: could not decode 'initial_date' ({e}); "
                      f"falling back to filename/--init-time value.", file=sys.stderr)

        return leadtime_hours, "lead_time", init_time

    raise KeyError(
        "Could not find a recognized time coordinate. Looked for 'time' or "
        f"'lead_time'+'initial_date'. Available variables: {list(ds.variables.keys())}"
    )


def extract_leadtime_station_slice(var, time_dim_name, station_indices):
    """
    Pulls a (n_leadtimes, n_stations) slice out of a variable regardless of
    its dimension order (handles (time, values) or
    (initial_date, lead_time, values)-style layouts).
    """
    dims = var.dimensions
    slicer = tuple(
        station_indices if d in STATION_DIM_NAMES else slice(None)
        for d in dims
    )
    raw = var[slicer]
    raw = np.ma.filled(raw, np.nan).astype(np.float32)

    time_axis = None
    squeeze_axes = []
    for i, d in enumerate(dims):
        if d == time_dim_name:
            time_axis = i
        elif d not in STATION_DIM_NAMES:
            if raw.shape[i] == 1:
                squeeze_axes.append(i)
            else:
                raise ValueError(
                    f"Variable '{var.name}' has an unexpected extra dimension "
                    f"'{d}' of size {raw.shape[i]}."
                )

    if time_axis is None:
        raise ValueError(f"Variable '{var.name}' has no '{time_dim_name}' dimension (dims={dims}).")

    for ax in sorted(squeeze_axes, reverse=True):
        raw = np.squeeze(raw, axis=ax)
        if ax < time_axis:
            time_axis -= 1

    if time_axis != 0:
        raw = np.moveaxis(raw, time_axis, 0)

    return raw


# =============================================================
# CORE: read one anemoi-style file -> (leadtimes, speed/dir per station)
# =============================================================

def extract_wind_from_anemoi_file(filepath, u_var, v_var, init_time):
    """
    Returns:
        leadtimes  : sorted list of float lead hours (valid_time - init_time)
        spd_array  : np.ndarray shape (n_leads, n_stations), float32, m/s
        dir_array  : np.ndarray shape (n_leads, n_stations), float32, degrees
    """
    ds = nc.Dataset(filepath, "r")

    if "latitude" not in ds.variables or "longitude" not in ds.variables:
        raise KeyError(f"'latitude'/'longitude' not found in {filepath}. "
                        f"Available: {list(ds.variables.keys())[:20]}")
    for v in (u_var, v_var):
        if v not in ds.variables:
            raise KeyError(f"'{v}' not found in {filepath}. "
                            f"Available: {[x for x in ds.variables if x not in ('time', 'latitude', 'longitude')]}")

    lat_arr = ds.variables["latitude"][:].astype("float64")
    lon_arr = ds.variables["longitude"][:].astype("float64")

    print(f"  Finding nearest grid points in {os.path.basename(filepath)} ...")
    station_indices = find_nearest_station_indices(lat_arr, lon_arr, STATIONS)

    leadtime_hours, time_dim_name, init_time = get_leadtimes_and_init_time(ds, init_time)

    if np.any(leadtime_hours < 0):
        print(f"  WARNING: {filepath} has {int(np.sum(leadtime_hours < 0))} step(s) with a "
              f"NEGATIVE leadtime relative to init_time={init_time} -- double check the "
              f"filename-parsed init time is correct for this file.", file=sys.stderr)

    # Pull the (leadtime, values) slice for just the stations we need, for
    # both u and v, regardless of the file's dimension convention.
    u_raw = extract_leadtime_station_slice(ds.variables[u_var], time_dim_name, station_indices)
    v_raw = extract_leadtime_station_slice(ds.variables[v_var], time_dim_name, station_indices)
    ds.close()

    spd, wdir = uv_to_speed_dir(u_raw, v_raw)

    order = np.argsort(leadtime_hours)
    sorted_leads = list(leadtime_hours[order])
    spd_array = spd[order, :]
    dir_array = wdir[order, :]
    return sorted_leads, spd_array, dir_array, init_time


# =============================================================
# WRITE / MERGE VERIF NETCDF  (unchanged logic from wrf2verif_wind.py)
# =============================================================

def write_verif_nc(output_file, init_time, variable, leadtimes, fcst_array):
    """
    Create or merge a VERIF-format NetCDF file.

    fcst_array shape: (n_leads, n_stations)
    leadtimes: sorted list of floats (hours)
    variable: "WSPD" or "WDIR" — controls units/long_name only.
    """
    n_locs = len(STATIONS)
    ids = np.array([s["id"] for s in STATIONS], dtype=np.int32)
    lats = np.array([s["lat"] for s in STATIONS], dtype=np.float32)
    lons = np.array([s["lon"] for s in STATIONS], dtype=np.float32)
    alts = np.array([s["alt"] for s in STATIONS], dtype=np.float32)

    init_unix = seconds_since_epoch(init_time)
    new_lead_arr = np.array(leadtimes, dtype=np.float32)

    if os.path.exists(output_file):
        print(f"  Output file exists — merging into {output_file} ...")

        with nc.Dataset(output_file, "r") as existing:
            ex_times = existing["time"][:]
            ex_leads = existing["leadtime"][:]
            ex_fcst = existing["fcst"][:]

        if init_unix in ex_times:
            print(f"  WARNING: init time {init_time} already in {output_file}. Overwriting that slot.")
            t_idx = int(np.where(ex_times == init_unix)[0][0])
            overwrite_slot = t_idx
        else:
            overwrite_slot = None

        merged_leads = np.union1d(ex_leads, new_lead_arr).astype(np.float32)
        n_leads_new = len(merged_leads)
        n_times_ex = len(ex_times)

        if overwrite_slot is not None:
            merged_times = ex_times.copy()
            n_times_out = n_times_ex
        else:
            merged_times = np.append(ex_times, init_unix).astype(np.int32)
            n_times_out = n_times_ex + 1

        merged_fcst = np.full((n_times_out, n_leads_new, n_locs), np.nan, dtype=np.float32)

        for old_li, old_lead in enumerate(ex_leads):
            new_li = int(np.where(merged_leads == old_lead)[0][0])
            for ti in range(n_times_ex):
                if overwrite_slot is not None and ti == overwrite_slot:
                    continue
                merged_fcst[ti, new_li, :] = ex_fcst[ti, old_li, :]

        new_t_idx = overwrite_slot if overwrite_slot is not None else n_times_ex
        for new_li_idx, lead_val in enumerate(new_lead_arr):
            ml_idx = int(np.where(merged_leads == lead_val)[0][0])
            merged_fcst[new_t_idx, ml_idx, :] = fcst_array[new_li_idx, :]

        os.remove(output_file)
        _write_nc_file(output_file, variable, merged_times, merged_leads,
                       ids, lats, lons, alts, merged_fcst)

    else:
        print(f"  Creating new output file: {output_file}")
        times_arr = np.array([init_unix], dtype=np.int32)
        fcst_3d = fcst_array[np.newaxis, :, :]
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
        out.createDimension("time", None)
        out.createDimension("leadtime", n_leads)
        out.createDimension("location", n_locs)

        v = out.createVariable("time", "i4", ("time",))
        v[:] = times_arr
        v.units = "seconds since 1970-01-01 00:00:00 +00:00"
        v.long_name = "Forecast initialization time"

        v = out.createVariable("leadtime", "f4", ("leadtime",))
        v[:] = leads_arr
        v.units = "hours"
        v.long_name = "Hours since forecast initialization"

        v = out.createVariable("location", "i4", ("location",))
        v[:] = ids
        v.long_name = "Station ID"

        v = out.createVariable("lat", "f4", ("location",))
        v[:] = lats
        v.units = "degrees_north"

        v = out.createVariable("lon", "f4", ("location",))
        v[:] = lons
        v.units = "degrees_east"

        v = out.createVariable("altitude", "f4", ("location",))
        v[:] = alts
        v.units = "meters"

        v = out.createVariable("obs", "f4", ("time", "leadtime", "location"), fill_value=np.nan)
        v[:] = np.full((n_times, n_leads, n_locs), np.nan, dtype=np.float32)
        v.long_name = "Observations (to be filled)"
        v.units = units

        v = out.createVariable("fcst", "f4", ("time", "leadtime", "location"), fill_value=np.nan)
        v[:] = fcst_3d
        v.long_name = f"anemoi {long_name} forecast"
        v.units = units
        v.source_variable = variable

        out.long_name = long_name
        out.standard_name = std_name
        out.units = units
        out.verif_version = "1.0.0"
        out.source = "anemoi/ECMWF-style model output"
        out.created_by = "anemoi2verif_wind.py"

    print(f"  Done → {output_file}  "
          f"(times={n_times}, leadtimes={n_leads}, locations={n_locs})")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert anemoi/ECMWF-style U/V output to Verif-format wind speed & direction NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("netcdf_files", nargs="+", help="Source file(s), one per init time, e.g. 20220701T00.nc")
    parser.add_argument("output_netcdf_file", help="Output base filename, e.g. output.nc "
                                                     "(writes output_wspd.nc and output_wdir.nc)")
    parser.add_argument("--level", default="10m", choices=LEVEL_CHOICES,
                         help="Which u/v pair to use: 10m (default, surface 10u/10v) "
                              "or a pressure level in hPa (50, 100, 250, 500, 850)")
    parser.add_argument("--init-time", default=None,
                         help="Force init time as YYYYMMDDHH instead of parsing it from the filename "
                              "(only sensible with a single input file)")

    args = parser.parse_args()
    u_var, v_var = uv_variable_names(args.level)

    if args.init_time and len(args.netcdf_files) > 1:
        print("WARNING: --init-time was given with multiple files -- every file will be "
              "tagged with the SAME init time, which is almost certainly not what you want "
              "unless every file really does share one init time.", file=sys.stderr)

    forced_init = parse_init_time(args.init_time) if args.init_time else None

    root, ext = os.path.splitext(args.output_netcdf_file)
    if not ext:
        ext = ".nc"
    spd_file = f"{root}_wspd{ext}"
    dir_file = f"{root}_wdir{ext}"

    print(f"\n{'='*60}")
    print(f"  anemoi -> Verif wind converter")
    print(f"  Level     : {args.level}  (u={u_var}, v={v_var})")
    print(f"  Files     : {len(args.netcdf_files)} file(s)")
    print(f"  Output    : {spd_file}, {dir_file}")
    print(f"{'='*60}\n")

    for fpath in sorted(args.netcdf_files):
        print(f"Processing {fpath} ...")
        if forced_init is not None:
            init_time = forced_init
        else:
            try:
                init_time = parse_init_time_from_filename(fpath)
            except ValueError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                sys.exit(1)

        leadtimes, spd_array, dir_array, init_time = extract_wind_from_anemoi_file(fpath, u_var, v_var, init_time)
        print(f"  Init time : {init_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Lead hours: {leadtimes}")

        write_verif_nc(spd_file, init_time, "WSPD", leadtimes, spd_array)
        write_verif_nc(dir_file, init_time, "WDIR", leadtimes, dir_array)

    print("\nDone!\n")


if __name__ == "__main__":
    main()