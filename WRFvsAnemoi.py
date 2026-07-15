#!/usr/bin/env python3
"""
Match a WRF output file (or set of WRF output files) against an ERA5-style
flat-"values" file, for ONE variable at a time, aligned on init time,
leadtime, and location.

KEY CHANGE FROM THE PREVIOUS VERSION: instead of picking a handful of
station points, this flattens WRF's native 2D grid (south_north x west_east)
into a 1D array with numpy .flatten() -- exactly mirroring the ERA5-style
file's flat "values" dimension. Every WRF grid cell becomes one candidate
"location". ERA5 (coarser, unstructured) is then nearest-neighbour sampled
onto each of those WRF points.

Because the two domains don't fully overlap, any WRF point whose nearest
ERA5 point is implausibly far away (i.e. it's outside ERA5's actual
coverage, so the "nearest" match is meaningless) is dropped before writing
output. The distance threshold is auto-estimated from ERA5's own median
point spacing unless you override it with --max-distance-km.

ROLES (fixed per your instruction):
  obs  = WRF   (the reference grid whose locations we flatten to)
  fcst = ERA5-style file, nearest-neighbour matched onto each WRF point

ALIGNMENT:
  - init time: WRF SIMULATION_START_DATE vs ERA5 epoch parsed from the
    `time` variable's units string
  - leadtime:  WRF = valid_time - init_time (from `Times`); ERA5 = time/3600
  - location:  WRF lat/lon flattened (numpy .flatten()); ERA5's nearest
    point to each flattened WRF cell found via KD-tree

NOTE ON SIZE: flattening a full WRF domain (426 x 303 here = 129,078 points)
into "locations" is a lot more than verif's format is typically used for
(station counts). It's still structurally valid netCDF, but the output
file will be large and any verif-side tooling that assumes "location" means
"a few dozen stations" may be slow. Use --max-points to subsample the
flattened array (deterministic stride, not random) if you want something
smaller while still testing the full pipeline.
"""

import argparse
import csv
import glob
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset, date2num


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between paired lat/lon arrays (broadcastable)."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bbox_prefilter(ref_lat, ref_lon, query_lat, query_lon, pad_deg=1.0):
    """Return indices of ref points that fall within the query points'
    lat/lon bounding box (+padding). This shrinks a huge unstructured
    reference grid (e.g. 342,916 ERA5 points) down to only the points
    near the region of interest, before any nearest-neighbour search --
    keeps brute-force distance computation cheap and avoids needing a
    spatial index (KD-tree) at all. Assumes a regional domain that
    doesn't cross the antimeridian."""
    lat_min, lat_max = query_lat.min() - pad_deg, query_lat.max() + pad_deg
    lon_min, lon_max = query_lon.min() - pad_deg, query_lon.max() + pad_deg
    mask = (ref_lat >= lat_min) & (ref_lat <= lat_max) & (ref_lon >= lon_min) & (ref_lon <= lon_max)
    return np.nonzero(mask)[0]


def nearest_neighbor_brute(query_lat, query_lon, ref_lat, ref_lon, max_chunk_elements=20_000_000):
    """For each query point, find the index (into ref_lat/ref_lon) and
    great-circle distance (km) of its nearest reference point, via plain
    numpy broadcasting -- no spatial index required. Processes queries in
    chunks so a (n_query x n_ref) distance matrix never gets too large."""
    n_query = len(query_lat)
    n_ref = len(ref_lat)
    nn_idx = np.empty(n_query, dtype=np.int64)
    nn_dist_km = np.empty(n_query, dtype=np.float64)

    chunk_size = max(1, max_chunk_elements // max(1, n_ref))
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        d = haversine_km(query_lat[start:end, None], query_lon[start:end, None],
                          ref_lat[None, :], ref_lon[None, :])  # (chunk, n_ref)
        idx_min = np.argmin(d, axis=1)
        nn_idx[start:end] = idx_min
        nn_dist_km[start:end] = d[np.arange(end - start), idx_min]
    return nn_idx, nn_dist_km


def estimate_era5_spacing_km(era5_lat, era5_lon, sample_size=1000):
    """Auto-estimate ERA5's typical point spacing: nearest-OTHER-point
    distance among a small random subsample (brute-force, no tree needed
    since the subsample itself is small)."""
    n = len(era5_lat)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sub_lat, sub_lon = era5_lat[idx], era5_lon[idx]
    d = haversine_km(sub_lat[:, None], sub_lon[:, None], sub_lat[None, :], sub_lon[None, :])
    np.fill_diagonal(d, np.inf)  # exclude self-match
    nearest_other = d.min(axis=1)
    return float(np.median(nearest_other))


def parse_wrf_time_string(char_array_row):
    chars = [c.decode() if isinstance(c, bytes) else c for c in char_array_row]
    s = "".join(chars)
    return datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")


def load_wrf(file_pattern, var_name):
    files = sorted(glob.glob(file_pattern)) if any(ch in file_pattern for ch in "*?[") else [file_pattern]
    if not files:
        print(f"ERROR: no WRF files matched '{file_pattern}'", file=sys.stderr)
        sys.exit(1)

    init_time = None
    lat2d = lon2d = None
    leadtimes, frames = [], []

    for fp in files:
        ds = Dataset(fp, "r")
        if init_time is None:
            init_time = datetime.strptime(ds.SIMULATION_START_DATE, "%Y-%m-%d_%H:%M:%S")
        if lat2d is None:
            lat2d = ds.variables["XLAT"][0, :, :].astype("float64")
            lon2d = ds.variables["XLONG"][0, :, :].astype("float64")
        if var_name not in ds.variables:
            avail = [v for v in ds.variables if ds.variables[v].dimensions[:1] == ("Time",)]
            print(f"ERROR: '{var_name}' not found in {fp}. Example available vars: {avail[:15]}", file=sys.stderr)
            sys.exit(1)

        times_char = ds.variables["Times"][:]
        n_t = times_char.shape[0]
        for t in range(n_t):
            valid_time = parse_wrf_time_string(times_char[t])
            leadtimes.append((valid_time - init_time).total_seconds() / 3600.0)
            frames.append(ds.variables[var_name][t, :, :].astype("float32"))
        ds.close()

    order = np.argsort(leadtimes)
    leadtime_hours = np.array(leadtimes, dtype="float64")[order]
    values = np.stack(frames, axis=0)[order]  # (leadtime, south_north, west_east)
    return init_time, leadtime_hours, lat2d, lon2d, values


def load_era5(fp, var_name):
    ds = Dataset(fp, "r")
    lat = ds.variables["latitude"][:].astype("float64")
    lon = ds.variables["longitude"][:].astype("float64")

    time_var = ds.variables["time"]
    epoch_str = time_var.units.split("since", 1)[1].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            epoch = datetime.strptime(epoch_str, fmt)
            break
        except ValueError:
            continue
    else:
        print(f"ERROR: could not parse ERA5 time units '{time_var.units}'", file=sys.stderr)
        sys.exit(1)

    leadtime_hours = time_var[:].astype("float64") / 3600.0

    if var_name not in ds.variables:
        avail = [v for v in ds.variables if v not in ("time", "latitude", "longitude")]
        print(f"ERROR: '{var_name}' not found in {fp}. Available: {avail}", file=sys.stderr)
        sys.exit(1)
    values = ds.variables[var_name][:, :].astype("float32")  # (time, values)
    ds.close()
    return epoch, leadtime_hours, lat, lon, values


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrf", required=True, help="WRF output file, or glob pattern e.g. 'wrfout_d02_*'")
    p.add_argument("--era5", required=True, help="ERA5-style .nc file")
    p.add_argument("--wrf-var", default="T2")
    p.add_argument("--era5-var", default="2t")
    p.add_argument("--tolerance-minutes", type=float, default=1.0, help="max time diff to call two leadtimes 'matched'")
    p.add_argument("--max-points", type=int, default=None,
                   help="optional: deterministically stride-subsample the flattened WRF grid to this many points (applied AFTER overlap filtering)")
    p.add_argument("--max-distance-km", type=float, default=None,
                   help="drop WRF points whose nearest ERA5 point is farther than this (km) -- i.e. outside ERA5's real coverage. "
                        "Default: auto-estimated as 1.5x ERA5's own median point spacing.")
    p.add_argument("--output-csv", default=None, help="optional CSV output (can be large -- omit for full-grid runs)")
    p.add_argument("--output-nc", default="verif_2t.nc", help="verif-format nc output path")
    args = p.parse_args()

    wrf_init, wrf_leadtime, wrf_lat, wrf_lon, wrf_vals = load_wrf(args.wrf, args.wrf_var)
    era5_init, era5_leadtime, era5_lat, era5_lon, era5_vals = load_era5(args.era5, args.era5_var)

    if wrf_init != era5_init:
        print(f"WARNING: init times differ -- WRF={wrf_init}, ERA5={era5_init}. "
              f"Leadtimes below are each relative to their OWN file's init time; "
              f"double check this is intentional.", file=sys.stderr)

    print(f"WRF init time (SIMULATION_START_DATE): {wrf_init}")
    print(f"ERA5 init time (parsed from `time` units): {era5_init}")
    print(f"WRF leadtimes (hours, {len(wrf_leadtime)} total):  {list(np.round(wrf_leadtime, 3))}")
    print(f"ERA5 leadtimes (hours, {len(era5_leadtime)} total): {list(np.round(era5_leadtime, 3))}")

    # --- match leadtimes between the two files ---
    tol_h = args.tolerance_minutes / 60.0
    matched_wrf_idx, matched_era5_idx, matched_leadtime = [], [], []
    for i, lt in enumerate(wrf_leadtime):
        j = np.argmin(np.abs(era5_leadtime - lt))
        if np.abs(era5_leadtime[j] - lt) <= tol_h:
            matched_wrf_idx.append(i)
            matched_era5_idx.append(j)
            matched_leadtime.append(lt)
    if not matched_leadtime:
        closest_diffs = [np.min(np.abs(era5_leadtime - lt)) for lt in wrf_leadtime]
        print("ERROR: no leadtimes matched within tolerance.", file=sys.stderr)
        print(f"       Tolerance was {tol_h * 60:.1f} minute(s). Closest ERA5 leadtime for each "
              f"WRF leadtime was off by (hours): {list(np.round(closest_diffs, 3))}", file=sys.stderr)
        print("       Common causes: (1) the two files' init times don't actually agree despite "
              "looking similar -- check the printed init times above; (2) ERA5's `time` units epoch "
              "isn't really the forecast init time (some datasets store *absolute* valid time there "
              "with an arbitrary reference epoch, not lead time -- in which case leadtime should be "
              "computed as (valid_time - init_time), same as WRF, not time/3600 directly); "
              "(3) a systematic constant offset (e.g. everything is off by exactly the same N hours) "
              "-- if the diffs above are all identical, that's your signal. "
              "Try --tolerance-minutes with a larger value to confirm it's an offset rather than "
              "a genuine non-overlap.", file=sys.stderr)
        sys.exit(1)
    print(f"Matched {len(matched_leadtime)} leadtime(s): {matched_leadtime}")

    # --- flatten WRF grid: this defines the (candidate) "location" dimension ---
    wrf_lat_flat = wrf_lat.flatten()
    wrf_lon_flat = wrf_lon.flatten()
    n_wrf_points_full = wrf_lat_flat.shape[0]
    print(f"Flattened WRF grid: {wrf_lat.shape} -> {n_wrf_points_full} points.")

    # --- nearest ERA5 point for every flattened WRF point, with distance ---
    # First shrink the (huge, unstructured) ERA5 point set down to just the
    # points near the WRF domain, so the brute-force search below stays cheap.
    era5_candidate_idx = bbox_prefilter(era5_lat, era5_lon, wrf_lat_flat, wrf_lon_flat, pad_deg=1.0)
    if len(era5_candidate_idx) == 0:
        print("ERROR: no ERA5 points fall anywhere near the WRF domain's bounding box -- "
              "the two files likely don't cover overlapping regions at all.", file=sys.stderr)
        sys.exit(1)
    era5_cand_lat = era5_lat[era5_candidate_idx]
    era5_cand_lon = era5_lon[era5_candidate_idx]

    nn_idx_local, dist_km_full = nearest_neighbor_brute(wrf_lat_flat, wrf_lon_flat, era5_cand_lat, era5_cand_lon)
    nn_idx_full = era5_candidate_idx[nn_idx_local]  # map back to indices into the full ERA5 array

    # --- keep only WRF points that actually fall within ERA5's coverage ---
    if args.max_distance_km is not None:
        threshold_km = args.max_distance_km
    else:
        threshold_km = 1.5 * estimate_era5_spacing_km(era5_lat, era5_lon)
    overlap_mask = dist_km_full <= threshold_km
    overlap_idx = np.nonzero(overlap_mask)[0]
    print(f"Overlap threshold: {threshold_km:.2f} km. "
          f"{len(overlap_idx)} / {n_wrf_points_full} WRF points fall within ERA5's coverage "
          f"({100 * len(overlap_idx) / n_wrf_points_full:.1f}%).")
    if len(overlap_idx) == 0:
        print("ERROR: no overlap found -- check that the two domains actually cover the same region.", file=sys.stderr)
        sys.exit(1)

    # --- optional further subsampling on top of the overlap region ---
    if args.max_points and args.max_points < len(overlap_idx):
        stride = max(1, len(overlap_idx) // args.max_points)
        keep_idx = overlap_idx[::stride][: args.max_points]
    else:
        keep_idx = overlap_idx

    loc_lat = wrf_lat_flat[keep_idx]
    loc_lon = wrf_lon_flat[keep_idx]
    loc_ids = keep_idx + 1  # 1-based location ids matching flattened WRF grid index
    n_loc = len(loc_ids)
    era5_nn_idx = nn_idx_full[keep_idx]
    print(f"Using {n_loc} overlapping location(s).")

    n_lt = len(matched_leadtime)
    obs_matched = np.full((n_lt, n_loc), np.nan, dtype="float32")   # WRF, flattened
    fcst_matched = np.full((n_lt, n_loc), np.nan, dtype="float32")  # ERA5, nearest-neighbour onto WRF points

    for k, (wi, ei) in enumerate(zip(matched_wrf_idx, matched_era5_idx)):
        wrf_frame_flat = wrf_vals[wi].flatten()          # numpy .flatten() as requested
        obs_matched[k, :] = wrf_frame_flat[keep_idx]
        era5_frame = era5_vals[ei]                       # already flat (time, values)
        fcst_matched[k, :] = era5_frame[era5_nn_idx]

    # --- optional CSV ---
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["leadtime_hour", "location_id", "lat", "lon",
                              "obs_wrf_" + args.wrf_var, "fcst_era5_" + args.era5_var])
            for k, lt in enumerate(matched_leadtime):
                for m in range(n_loc):
                    writer.writerow([lt, loc_ids[m], loc_lat[m], loc_lon[m],
                                      obs_matched[k, m], fcst_matched[k, m]])
        print(f"Wrote {args.output_csv}")

    # --- verif-format nc: obs = WRF, fcst = ERA5 ---
    out = Dataset(args.output_nc, "w", format="NETCDF4")
    out.createDimension("time", None)
    out.createDimension("leadtime", n_lt)
    out.createDimension("location", n_loc)

    v_time = out.createVariable("time", "i4", ("time",))
    v_time.units = "seconds since 1970-01-01 00:00:00 +00:00"
    v_time[:] = date2num([wrf_init], units=v_time.units)

    v_leadtime = out.createVariable("leadtime", "f4", ("leadtime",))
    v_leadtime.units = "hour"
    v_leadtime[:] = np.array(matched_leadtime)

    v_location = out.createVariable("location", "i4", ("location",))
    v_location[:] = loc_ids

    v_lat = out.createVariable("lat", "f4", ("location",))
    v_lat.units = "degrees_north"
    v_lat[:] = loc_lat

    v_lon = out.createVariable("lon", "f4", ("location",))
    v_lon.units = "degrees_east"
    v_lon[:] = loc_lon

    v_obs = out.createVariable("obs", "f4", ("time", "leadtime", "location"), fill_value=np.nan)
    v_obs[0, :, :] = obs_matched

    v_fcst = out.createVariable("fcst", "f4", ("time", "leadtime", "location"), fill_value=np.nan)
    v_fcst[0, :, :] = fcst_matched

    out.long_name = "2 metre temperature"
    out.standard_name = "air_temperature"
    out.verif_version = "1.0.0"
    out.close()
    print(f"Wrote {args.output_nc} (obs=WRF {args.wrf_var}, fcst=ERA5 {args.era5_var}, "
          f"{n_lt} leadtime(s) x {n_loc} location(s))")


if __name__ == "__main__":
    main()