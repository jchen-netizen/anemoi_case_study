#!/usr/bin/env python3
"""
wrf2verif_rh2.py — Extract WRF 2-m relative humidity (rh2) point forecasts
into Verif-format NetCDF.

Uses wrf-python's getvar(..., "rh2") to compute relative humidity the same
way wrf-python does internally (from T2, Q2, PSFC), rather than reading a
stored variable — rh2 is a diagnostic, not a raw field in wrfout files.

Usage:
    python3 wrf2verif_rh2.py <wrf_files> <init_time> <output.nc>

Arguments:
    wrf_files       One or more WRF output files (glob-expanded by shell, e.g. wrfout_d02_*)
    init_time       Forecast initialization time as YYYYMMDDHH  (e.g. 2023050800)
    output.nc       Output Verif-format NetCDF file

Example:
    python3 wrf2verif_rh2.py wrfout_d02_2023-05-08_* 2023050800 rh2_output.nc

Behaviour:
    - If output.nc already exists, the new init time is MERGED in
      (appended along the time dimension).
    - Each WRF file contributes ONE valid time → ONE leadtime step.
    - Leadtime is derived from the filename timestamp minus init_time (hours).
    - Observations are left as NaN (fill in later).
    - Files are read directly from their original location (no local copies
      are made, to avoid using disk space). Flaky reads over network-mounted
      volumes are retried a few times with a short delay before giving up
      on that file.

Stations (hardcoded — add more to STATIONS list):
    id=129  Pink Mountain  lat=56.94  lon=-122.70  alt=960.10 m
    id=120  Silver        lat=57.37  lon=-121.41  alt=835.00 m
    id=131  Muskwa        lat=57.88  lon=-123.62  alt=769.00 m
"""

import os
# Must be set before netCDF4/wrf-python touch HDF5 — avoids locking issues
# on network-mounted volumes (SMB/AFP/NFS).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
import re
import time
import argparse
import numpy as np
import netCDF4 as nc
from datetime import datetime, timezone
from wrf import getvar

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


def is_wrf_output_candidate(filepath):
    """Return True only for filenames that look like real WRF output timestamps ending in HH:MM:SS."""
    basename = os.path.basename(filepath)
    if not basename:
        return False
    if "merge" in basename.lower():
        return False
    if basename.endswith((".OK", ".ok", ".txt", ".nc")):
        return False
    return bool(re.search(r'\d{2}:\d{2}:\d{2}$', basename))


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

    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})[:_](\d{2})[:_](\d{2})', basename)
    if m:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)

    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_T](\d{6})', basename)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = m.group(4)
        h, mi, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)

    raise ValueError(f"Cannot parse valid time from filename: {basename}")


def find_nearest_idx(ncfile, target_lat, target_lon):
    """Return (sn_idx, we_idx) of nearest WRF grid point, using getvar's XLAT/XLONG."""
    xlat = getvar(ncfile, "XLAT").values
    xlong = getvar(ncfile, "XLONG").values

    dist = np.sqrt((xlat - target_lat) ** 2 + (xlong - target_lon) ** 2)
    min_idx = np.unravel_index(dist.argmin(), dist.shape)
    sn_idx, we_idx = int(min_idx[0]), int(min_idx[1])
    return sn_idx, we_idx


def seconds_since_epoch(dt):
    return int((dt - EPOCH).total_seconds())


def open_nc_with_retry(fpath, attempts=3, delay=2.0):
    """
    Open fpath directly with netCDF4, retrying a few times on failure.
    Returns an open nc.Dataset, or None if all attempts fail.
    No local copy is made — reads happen straight from fpath.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return nc.Dataset(fpath)
        except Exception as e:
            last_err = e
            if attempt < attempts:
                time.sleep(delay)
    print(f"  WARNING: failed to open {fpath} after {attempts} attempts ({last_err}) — skipping.")
    return None


# =============================================================
# CORE: read WRF files → (leadtimes, fcst values per station)
# =============================================================

def extract_rh2_from_wrf_files(wrf_files, init_time):
    """
    Returns:
        leadtimes  : sorted list of int lead hours
        fcst_array : np.ndarray shape (n_leads, n_stations), float32, percent RH
    """
    n_stations = len(STATIONS)
    sorted_files = sorted(
        [f for f in wrf_files if is_wrf_output_candidate(f)],
        key=lambda x: os.path.basename(x),
    )

    if not sorted_files:
        raise RuntimeError("No matching WRF output files found. Expected filenames ending in HH:MM:SS and not containing 'merge'.")

    # --- Find nearest grid indices using the first readable file ---------------
    station_indices = None
    for fpath in sorted_files:
        ncfile0 = open_nc_with_retry(fpath)
        if ncfile0 is None:
            continue
        try:
            print(f"  Finding nearest grid points using {fpath} ...")
            station_indices = []
            for st in STATIONS:
                sn, we = find_nearest_idx(ncfile0, st["lat"], st["lon"])
                xlat = getvar(ncfile0, "XLAT").values
                xlong = getvar(ncfile0, "XLONG").values
                print(f"    {st['name']:20s}  target=({st['lat']:.4f},{st['lon']:.4f})"
                      f"  nearest=({xlat[sn, we]:.4f},{xlong[sn, we]:.4f})  grid=({sn},{we})")
                station_indices.append((sn, we))
        except Exception as e:
            print(f"  WARNING: failed to read {fpath} ({e}) — trying next file.")
            station_indices = None
        finally:
            ncfile0.close()

        if station_indices is not None:
            break

    if station_indices is None:
        raise RuntimeError("Could not find station grid indices — no readable WRF file.")

    # --- Read each file → one leadtime step -------------------------------------
    records = {}  # lead_hour (int) → np.array shape (n_stations,)

    for fpath in sorted_files:
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

        ncfile = open_nc_with_retry(fpath)
        if ncfile is None:
            continue

        try:
            rh2 = getvar(ncfile, "rh2", timeidx=0)
        except Exception as e:
            print(f"  WARNING: failed to compute rh2 for {fpath} ({e}) — skipping.")
            ncfile.close()
            continue

        vals = np.full(n_stations, np.nan, dtype=np.float32)
        for i, (sn, we) in enumerate(station_indices):
            vals[i] = float(rh2[sn, we].values)

        print(f"  {os.path.basename(fpath)}  valid={valid_time.strftime('%Y-%m-%d %H:%M')}  lead={lead_int}h")
        records[lead_int] = vals

        ncfile.close()

    if not records:
        raise RuntimeError("No valid WRF files could be processed.")

    sorted_leads = sorted(records.keys())
    fcst_array = np.stack([records[l] for l in sorted_leads], axis=0)  # (n_leads, n_stations)
    return sorted_leads, fcst_array


# =============================================================
# WRITE / MERGE VERIF NETCDF
# =============================================================

def write_verif_nc(output_file, init_time, leadtimes, fcst_array):
    """
    Create or merge a VERIF-format NetCDF file for rh2 (percent).

    fcst_array shape: (n_leads, n_stations)
    leadtimes: sorted list of ints (hours)
    """
    n_locs = len(STATIONS)
    ids  = np.array([s["id"]  for s in STATIONS], dtype=np.int32)
    lats = np.array([s["lat"] for s in STATIONS], dtype=np.float32)
    lons = np.array([s["lon"] for s in STATIONS], dtype=np.float32)
    alts = np.array([s["alt"] for s in STATIONS], dtype=np.float32)

    init_unix = seconds_since_epoch(init_time)
    new_lead_arr = np.array(leadtimes, dtype=np.float32)

    if os.path.exists(output_file):
        # === MERGE mode =========================================================
        print(f"  Output file exists — merging into {output_file} ...")

        with nc.Dataset(output_file, "r") as existing:
            ex_times = existing["time"][:]
            ex_leads = existing["leadtime"][:]
            ex_fcst  = existing["fcst"][:]

        if init_unix in ex_times:
            print(f"  WARNING: init time {init_time} already in {output_file}. Overwriting that slot.")
            overwrite_slot = int(np.where(ex_times == init_unix)[0][0])
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
        _write_nc_file(output_file, merged_times, merged_leads, ids, lats, lons, alts, merged_fcst)

    else:
        # === CREATE mode =========================================================
        print(f"  Creating new output file: {output_file}")
        times_arr = np.array([init_unix], dtype=np.int32)
        fcst_3d = fcst_array[np.newaxis, :, :]
        _write_nc_file(output_file, times_arr, new_lead_arr, ids, lats, lons, alts, fcst_3d)


def _write_nc_file(output_file, times_arr, leads_arr, ids, lats, lons, alts, fcst_3d):
    """Low-level writer — always creates a fresh file."""
    n_times, n_leads, n_locs = fcst_3d.shape

    with nc.Dataset(output_file, "w", format="NETCDF4") as out:

        out.createDimension("time", None)      # UNLIMITED
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
        v.units = "percent"

        v = out.createVariable("fcst", "f4", ("time", "leadtime", "location"), fill_value=np.nan)
        v[:] = fcst_3d
        v.long_name = "WRF rh2 forecast"
        v.units = "percent"
        v.wrf_variable = "rh2"

        out.long_name = "Relative Humidity"
        out.standard_name = "relative_humidity"
        out.units = "percent"
        out.verif_version = "1.0.0"
        out.source = "WRF model output (rh2 via wrf-python getvar)"
        out.created_by = "wrf_rh2_to_verif.py"

    print(f"  Done → {output_file}  "
          f"(times={n_times}, leadtimes={n_leads}, locations={n_locs})")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert WRF output files to Verif-format NetCDF (rh2, via wrf-python getvar).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("netcdf_files", nargs="+", help="WRF output file(s)")
    parser.add_argument("initialized_time", help="Init time as YYYYMMDDHH (e.g. 2023050800)")
    parser.add_argument("output_netcdf_file", help="Output Verif-format NetCDF filename")

    args = parser.parse_args()

    wrf_files = args.netcdf_files
    init_time = parse_init_time(args.initialized_time)
    output_file = args.output_netcdf_file

    print(f"\n{'='*60}")
    print(f"  WRF rh2 → Verif converter")
    print(f"  Init time : {init_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Variable  : rh2 (via wrf-python getvar)")
    print(f"  Files     : {len(wrf_files)} file(s)")
    print(f"  Output    : {output_file}")
    print(f"{'='*60}\n")

    print("Step 1: Extracting rh2 from WRF files ...")
    leadtimes, fcst_array = extract_rh2_from_wrf_files(wrf_files, init_time)
    print(f"  → Lead hours: {leadtimes}")

    print("\nStep 2: Writing Verif-format NetCDF ...")
    write_verif_nc(output_file, init_time, leadtimes, fcst_array)

    print("\nDone!\n")


if __name__ == "__main__":
    main()