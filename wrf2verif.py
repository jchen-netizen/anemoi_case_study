#!/usr/bin/env python3
"""
wrf2verif.py — Extract WRF point forecasts into Verif-format NetCDF.

Usage:
    python3 wrf2verif.py <wrf_files> <init_time> <variable> <output.nc>

Arguments:
    wrf_files       One or more WRF output files (glob-expanded by shell, e.g. wrfout_d02_*)
    init_time       Forecast initialization time as YYYYMMDDHH  (e.g. 2023050800)
    variable        WRF variable name to extract               (e.g. T2)
    output.nc       Output Verif-format NetCDF file

Examples:
    python3 wrf2verif.py wrfout_d02_* 2023050800 T2 output.nc
    python3 wrf2verif.py wrfout_d02_2023-05-08_* 2023050800 T2 output.nc

Behaviour:
    - If output.nc already exists AND contains the same variable, the new
      init time is MERGED in (appended along the time dimension).
    - Each WRF file contributes ONE valid time → ONE leadtime step.
    - Leadtime is derived from the filename timestamp minus init_time (hours).
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
    """Return (sn_idx, we_idx) of nearest WRF grid point."""
    xlat  = ds["XLAT"].values   # shape: (Time, south_north, west_east)
    xlong = ds["XLONG"].values

    # Use first time slice for the lat/lon grid
    dist = np.sqrt((xlat[0] - target_lat)**2 + (xlong[0] - target_lon)**2)
    min_idx = np.unravel_index(dist.argmin(), dist.shape)
    sn_idx, we_idx = int(min_idx[0]), int(min_idx[1])
    return sn_idx, we_idx


def seconds_since_epoch(dt):
    return int((dt - EPOCH).total_seconds())


# =============================================================
# CORE: read WRF files → (leadtimes, fcst values per station)
# =============================================================

def extract_from_wrf_files(wrf_files, init_time, variable):
    """
    Returns:
        leadtimes  : sorted list of float lead hours
        fcst_array : np.ndarray shape (n_leads, n_stations), float32, °C if T2
    """
    n_stations = len(STATIONS)

    # Pre-find nearest grid indices using the first file
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
    records = {}  # lead_hour (int) → np.array shape (n_stations,)

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
        vals = np.full(n_stations, np.nan, dtype=np.float32)

        for i, (sn, we) in enumerate(station_indices):
            raw = float(ds[variable].isel(south_north=sn, west_east=we).values.flat[0])
            # Convert Kelvin → Celsius for temperature variables
            if variable in ("T2", "T", "TSK", "TH2"):
                raw -= 273.15
            vals[i] = raw

        records[lead_int] = vals
        ds.close()

    if not records:
        raise RuntimeError("No valid WRF files could be processed.")

    sorted_leads = sorted(records.keys())
    fcst_array = np.stack([records[l] for l in sorted_leads], axis=0)  # (n_leads, n_stations)
    return sorted_leads, fcst_array


# =============================================================
# WRITE / MERGE VERIF NETCDF
# =============================================================

def write_verif_nc(output_file, init_time, variable, leadtimes, fcst_array):
    """
    Create or merge a VERIF-format NetCDF file.

    fcst_array shape: (n_leads, n_stations)
    leadtimes: sorted list of ints (hours)
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
        v.units     = "celsius" if variable in ("T2", "T", "TSK", "TH2") else "unknown"

        # fcst
        v = out.createVariable("fcst", "f4", ("time", "leadtime", "location"),
                               fill_value=np.nan)
        v[:] = fcst_3d
        v.long_name = f"WRF {variable} forecast"
        v.units     = "celsius" if variable in ("T2", "T", "TSK", "TH2") else "unknown"
        v.wrf_variable = variable

        # Global attributes (Verif standard)
        out.long_name     = "Temperature" if variable in ("T2", "T", "TSK", "TH2") else variable
        out.standard_name = "air_temperature" if variable in ("T2", "T", "TSK", "TH2") else variable
        out.units         = "celsius" if variable in ("T2", "T", "TSK", "TH2") else "unknown"
        out.verif_version = "1.0.0"
        out.source        = "WRF model output"
        out.created_by    = "wrf_to_verif.py"

    print(f"  Done → {output_file}  "
          f"(times={n_times}, leadtimes={n_leads}, locations={n_locs})")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert WRF output files to Verif-format NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("netcdf_files",   nargs="+", help="WRF output file(s)")
    parser.add_argument("initialized_time", help="Init time as YYYYMMDDHH (e.g. 2023050800)")
    parser.add_argument("variable",       help="WRF variable name (e.g. T2)")
    parser.add_argument("output_netcdf_file", help="Output Verif-format NetCDF filename")

    args = parser.parse_args()

    wrf_files   = args.netcdf_files
    init_time   = parse_init_time(args.initialized_time)
    variable    = args.variable
    output_file = args.output_netcdf_file

    print(f"\n{'='*60}")
    print(f"  WRF → Verif converter")
    print(f"  Init time : {init_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Variable  : {variable}")
    print(f"  Files     : {len(wrf_files)} file(s)")
    print(f"  Output    : {output_file}")
    print(f"{'='*60}\n")

    print("Step 1: Extracting from WRF files ...")
    leadtimes, fcst_array = extract_from_wrf_files(wrf_files, init_time, variable)
    print(f"  → Lead hours: {leadtimes}")

    print("\nStep 2: Writing Verif-format NetCDF ...")
    write_verif_nc(output_file, init_time, variable, leadtimes, fcst_array)

    print("\nDone!\n")


if __name__ == "__main__":
    main()
