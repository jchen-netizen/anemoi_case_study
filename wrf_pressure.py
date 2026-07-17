#!/usr/bin/env python3
"""
wrf_pressure.py

Reads WRF output files (wrfout_d0X_*) organized in per-initialization-date
folders (e.g. 23010100/, 23010200/, 23050800/, ...), computes Mean Sea Level
Pressure (MSLP) and Surface Pressure (SP) using wrf-python, converts both to
kPa, and writes a copy of each input file with the two new variables added
into a matching folder under the output directory.

Usage:
    python3 wrf_pressure.py <input_dir> <out_dir>

<input_dir> can be either:
    1) A single date folder that directly contains wrfout_* files, e.g.:
           python3 wrf_pressure.py 23050800 out_dir
       -> out_dir/23050800/wrfout_d02_..._pressure.nc, one per input file

    2) A parent folder containing multiple date subfolders, e.g.:
           parent/
             23010100/wrfout_d02_...
             23010200/wrfout_d02_...
             23050800/wrfout_d02_...
       Run as:
           python3 wrf_pressure.py parent out_dir
       -> out_dir/23010100/wrfout_d02_..._pressure.nc
          out_dir/23010200/wrfout_d02_..._pressure.nc
          out_dir/23050800/wrfout_d02_..._pressure.nc
       i.e. one matching output subfolder per date folder found.

Each output file keeps every original variable from its input file, plus
MSLP and SP in kPa. Output filenames are "<original_filename>_pressure.nc".

Actually might not be a good idea for storage but it's a later problem.
"""

import sys
import glob
import os
import re

import xarray as xr
from netCDF4 import Dataset
import wrf


WRFOUT_PATTERN = "wrfout*"


def parse_args(argv):
    if len(argv) != 2:
        sys.exit("Usage: python3 wrf_pressure.py <input_dir> <out_dir>")
    input_dir, out_dir = argv
    if not os.path.isdir(input_dir):
        sys.exit(f"Input directory not found: {input_dir}")
    return input_dir, out_dir


WRFOUT_REGEX = re.compile(r"^wrfout_d0[1-9]_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}$")


def find_wrfout_files(folder):
    return sorted(
        p for p in glob.glob(os.path.join(folder, WRFOUT_PATTERN))
        if WRFOUT_REGEX.match(os.path.basename(p))
    )


def find_date_folders(input_dir):
    """
    Returns a list of (date_folder_name, folder_path, [wrfout files]) tuples.

    If input_dir itself directly contains wrfout files, it is treated as a
    single date folder. Otherwise, its immediate subdirectories that contain
    wrfout files are each treated as a date folder.
    """
    direct_files = find_wrfout_files(input_dir)
    if direct_files:
        name = os.path.basename(os.path.normpath(input_dir))
        return [(name, input_dir, direct_files)]

    date_folders = []
    for entry in sorted(os.listdir(input_dir)):
        sub_path = os.path.join(input_dir, entry)
        if os.path.isdir(sub_path):
            files = find_wrfout_files(sub_path)
            if files:
                date_folders.append((entry, sub_path, files))

    if not date_folders:
        sys.exit(f"No wrfout files found in {input_dir} or its subfolders")

    return date_folders


def strip_coords(da):
    return xr.DataArray(da.values, dims=da.dims, attrs=dict(da.attrs))


def process_file(input_file, out_folder):
    print(f"  Processing {input_file} ...")

    nc = Dataset(input_file)
    try:
        slp = wrf.getvar(nc, "slp", timeidx=wrf.ALL_TIMES)
        psfc = wrf.getvar(nc, "PSFC", timeidx=wrf.ALL_TIMES)
    finally:
        nc.close()

    # slp from wrf-python is in hPa -> kPa
    mslp_kpa = slp * 0.1
    mslp_kpa.name = "MSLP"
    mslp_kpa.attrs["units"] = "kPa"
    mslp_kpa.attrs["description"] = "Mean Sea Level Pressure"

    # PSFC from wrf-python/WRF is in Pa -> kPa
    sp_kpa = psfc * 0.001
    sp_kpa.name = "SP"
    sp_kpa.attrs["units"] = "kPa"
    sp_kpa.attrs["description"] = "Surface Pressure"

    ds = xr.open_dataset(
        input_file, decode_times=False, decode_coords=False, mask_and_scale=False
    )
    ds.load()

    ds["MSLP"] = strip_coords(mslp_kpa)
    ds["SP"] = strip_coords(sp_kpa)

    out_file = os.path.join(out_folder, f"{os.path.basename(input_file)}_pressure.nc")
    print(f"    -> {out_file}")
    ds.to_netcdf(out_file, format="NETCDF4")
    ds.close()


def main():
    input_dir, out_dir = parse_args(sys.argv[1:])
    date_folders = find_date_folders(input_dir)

    print(f"Found {len(date_folders)} date folder(s):")
    for name, path, files in date_folders:
        print(f"  {name} ({path}): {len(files)} file(s)")

    for name, path, files in date_folders:
        out_folder = os.path.join(out_dir, name)
        os.makedirs(out_folder, exist_ok=True)
        print(f"Processing date folder: {name}")
        for f in files:
            process_file(f, out_folder)

    print("Done.")


if __name__ == "__main__":
    main()