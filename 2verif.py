#!/usr/bin/env python3
"""
============================================================
 Verif-format converter: WRF / Anemoi (BC) / Anemoi (Norway) / Anemoi (Global) vs ClimatEx (truth)
============================================================
Produces 4 verif-ready NetCDF files (one per forecast source).
Each output file uses that forecast source's own native grid as
the `location` dimension. Uses nearest-neighbor resampled onto each
forecast's grid to produce the matching `obs` values.

How to use:
1. Edit the CONFIG block below
2. Run: python make_verif.py
3. To add another init date: change DATE, re-run.

Behaviour: 
- One variable is processed per run
- leadtime index 0..14 == valid hours 0,6,...,84 for ALL FOUR
  forecast sources.
============================================================
"""
 
import os
import datetime
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, latlon_coords
 
# ============================================================
# CONFIG -- EDIT THIS BLOCK FOR EACH RUN
# ============================================================
 
# ---- init date for this run ----
DATE = datetime.datetime(2023, 5, 8)
 
# ---- leadtimes: index 0..14 -> hours 0,6,...,84 ----
LEADTIME_INDICES = list(range(15))
LEADTIME_HOURS = [i * 6 for i in LEADTIME_INDICES]
 
# ---- variable to extract from each source + unit fix ----
# final_value = raw_value * scale + offset
VARIABLES = {
    'wrf':    {'name': 'T2', 'scale': 1.0, 'offset': -273.15},
    'cx':     {'name': 'T2', 'scale': 1.0, 'offset': -273.15},
    'bc':     {'name': '2t', 'scale': 1.0, 'offset': -273.15},
    'norway': {'name': '2t', 'scale': 1.0, 'offset': -273.15},
    'global': {'name': '2t', 'scale': 1.0, 'offset': -273.15},
}
 
LONG_NAME = "Temperature"
STANDARD_NAME = "air_temperature"
 
# ---- path builders (mirror the patterns you gave) ----
def wrf_path(date, valid_dt):
    return (f"/Volumes/jchen/Share_Forecasts/WAC00WG-01/"
            f"{date.strftime('%y%m%d')}00/"
            f"wrfout_d02_{valid_dt.strftime('%Y-%m-%d_%H:00:00')}")
 
def bc_path(date):
    return f"/Users/jchen/Anemoi/ignition/bc/{date.strftime('%Y%m%d')}T00.nc"
 
def norway_path(date):
    return f"/Users/jchen/Anemoi/ignition/norway/{date.strftime('%Y%m%d')}T00.nc"
 
def global_path(date):
    return f"/Volumes/jchen/Share_anemoi/results/donnie_creek/global-model/{date.strftime('%Y%m%d')}T00.nc"
 
def cx_path(valid_dt):
    return (f"/Volumes/Scratch/jchen-scratch/ClimatEx/compressed-3d-wrfout/"
            f"wrfout_d03_{valid_dt.strftime('%Y-%m-%d_%H:00:00')}_compressed")
 
# ---- output files ----
OUTPUT_DIR = "/Users/jchen/Anemoi/verif_output"
OUTPUT_FILES = {
    'wrf':    os.path.join(OUTPUT_DIR, "verif_wrf.nc"),
    'bc':     os.path.join(OUTPUT_DIR, "verif_bc.nc"),
    'norway': os.path.join(OUTPUT_DIR, "verif_norway.nc"),
    'global': os.path.join(OUTPUT_DIR, "verif_global.nc"),
}
 
# ============================================================
# HELPERS
# ============================================================
 
def apply_units(data, varcfg):
    return data * varcfg['scale'] + varcfg['offset']
 
 
def load_wrf_like(path, varname):
    """For wrf-format files (wrf + cx). Returns (data2d, lat2d, lon2d)."""
    ds = Dataset(path)
    var = getvar(ds, varname)
    data = np.asarray(var.values)
    lat, lon = latlon_coords(var)
    ds.close()
    return data, np.asarray(lat), np.asarray(lon)
 
 
def load_anemoi_leadtime(xr_ds, leadtime_index, varname):
    """For bc/norway/global xarray datasets. Returns (data2d, lat2d, lon2d)."""
    tds = xr_ds.sel(lead_time=xr_ds.lead_time[leadtime_index])
    data = np.asarray(tds[varname].values)
    lat = np.asarray(tds['latitude'].values)
    lon = np.asarray(tds['longitude'].values)
    return data, lat, lon
 
 
def nearest_obs_on_grid(obs_lat, obs_lon, obs_data, target_lat, target_lon):
    """
    Nearest-neighbor resample obs (bigger domain) onto a forecast's grid,
    using only xarray/numpy (no scipy).
 
    Broadcasts every obs point against every target point along two
    separate dims, finds the closest obs point per target point with
    argmin, then gathers the matching obs values. This builds an
    (n_target, n_obs) array in memory -- fine for typical WRF-sized
    domains, but if your obs grid is very large and this gets slow /
    memory-heavy, chunk it with dask (`.chunk(...)`) before argmin.
    """
    obs_lat_da = xr.DataArray(obs_lat.ravel(), dims="obs_pts")
    obs_lon_da = xr.DataArray(obs_lon.ravel(), dims="obs_pts")
    obs_data_da = xr.DataArray(obs_data.ravel(), dims="obs_pts")
 
    tgt_lat_da = xr.DataArray(target_lat.ravel(), dims="tgt_pts")
    tgt_lon_da = xr.DataArray(target_lon.ravel(), dims="tgt_pts")
 
    # squared distance in lat/lon degrees, broadcasts to (tgt_pts, obs_pts)
    dist2 = (obs_lat_da - tgt_lat_da) ** 2 + (obs_lon_da - tgt_lon_da) ** 2
 
    nearest_idx = dist2.argmin(dim="obs_pts")
    matched = obs_data_da.isel(obs_pts=nearest_idx)
 
    return matched.values.reshape(target_lat.shape)
 
 
def init_epoch_seconds(dt):
    return int((dt - datetime.datetime(1970, 1, 1)).total_seconds())
 
 
def write_verif_file(filepath, init_dt, leadtime_hours, lat2d, lon2d,
                      obs_2d_by_leadtime, fcst_2d_by_leadtime,
                      long_name, standard_name):
    """
    Create or append-to a verif-format NetCDF file.
    `location` = flattened native grid of this forecast source
    (assumed static across different init-date runs).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
 
    n_leadtime = len(leadtime_hours)
    lat_flat = lat2d.ravel().astype('float32')
    lon_flat = lon2d.ravel().astype('float32')
    n_loc = lat_flat.size
 
    init_sec = init_epoch_seconds(init_dt)
 
    file_exists = os.path.exists(filepath)
    ds = Dataset(filepath, 'a' if file_exists else 'w')
 
    if not file_exists:
        ds.createDimension('time', None)
        ds.createDimension('leadtime', n_leadtime)
        ds.createDimension('location', n_loc)
 
        v_time = ds.createVariable('time', 'i4', ('time',))
        v_leadtime = ds.createVariable('leadtime', 'f4', ('leadtime',))
        v_location = ds.createVariable('location', 'i4', ('location',))
        v_lat = ds.createVariable('lat', 'f4', ('location',))
        v_lon = ds.createVariable('lon', 'f4', ('location',))
        v_obs = ds.createVariable('obs', 'f4', ('time', 'leadtime', 'location'),
                                   fill_value=np.nan)
        v_fcst = ds.createVariable('fcst', 'f4', ('time', 'leadtime', 'location'),
                                    fill_value=np.nan)
 
        v_leadtime[:] = np.array(leadtime_hours, dtype='float32')
        v_location[:] = np.arange(n_loc, dtype='int32')
        v_lat[:] = lat_flat
        v_lon[:] = lon_flat
 
        ds.long_name = long_name
        ds.standard_name = standard_name
        ds.verif_version = "1.0.0"
 
        t_index = 0
    else:
        v_time = ds.variables['time']
        v_obs = ds.variables['obs']
        v_fcst = ds.variables['fcst']
 
        existing_loc = ds.dimensions['location'].size
        if existing_loc != n_loc:
            ds.close()
            raise ValueError(
                f"{filepath}: existing location dim ({existing_loc}) != "
                f"this run's grid size ({n_loc}). The forecast grid must "
                f"stay the same across runs for this file."
            )
 
        existing_times = v_time[:].tolist()
        t_index = existing_times.index(init_sec) if init_sec in existing_times \
            else len(existing_times)
 
    v_time[t_index] = init_sec
 
    obs_stack = np.stack([o.ravel() for o in obs_2d_by_leadtime], axis=0)
    fcst_stack = np.stack([f.ravel() for f in fcst_2d_by_leadtime], axis=0)
    v_obs[t_index, :, :] = obs_stack
    v_fcst[t_index, :, :] = fcst_stack
 
    ds.close()
    print(f"Wrote {filepath}  (time index {t_index}, init={init_dt}, "
          f"{n_leadtime} leadtimes, {n_loc} locations)")
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    # sources where "one file has all leadtimes" -- open once
    bc_ds = xr.open_dataset(bc_path(DATE), engine='netcdf4')
    norway_ds = xr.open_dataset(norway_path(DATE), engine='netcdf4')
    global_ds = xr.open_dataset(global_path(DATE), engine='netcdf4')
 
    sources = ['wrf', 'bc', 'norway', 'global']
    fcst_by_source = {k: [] for k in sources}
    obs_by_source = {k: [] for k in sources}
    grid_by_source = {}  # captured once per source, assumed static grid
 
    for li in LEADTIME_INDICES:
        hours = li * 6
        valid_dt = DATE + datetime.timedelta(hours=hours)
 
        # ---- obs (cx): one file per valid time ----
        cx_var = VARIABLES['cx']
        obs_data, obs_lat, obs_lon = load_wrf_like(cx_path(valid_dt), cx_var['name'])
        obs_data = apply_units(obs_data, cx_var)
 
        # ---- wrf: hourly files, pick the one matching this valid time ----
        wrf_var = VARIABLES['wrf']
        try:
            w_data, w_lat, w_lon = load_wrf_like(wrf_path(DATE, valid_dt), wrf_var['name'])
            w_data = apply_units(w_data, wrf_var)
        except FileNotFoundError:
            print(f"[wrf] missing file for valid_dt={valid_dt}, filling NaN")
            if 'wrf' not in grid_by_source:
                raise  # can't recover shape if we've never seen this grid
            w_lat, w_lon = grid_by_source['wrf']
            w_data = np.full(w_lat.shape, np.nan, dtype='float32')
        grid_by_source.setdefault('wrf', (w_lat, w_lon))
        w_obs = nearest_obs_on_grid(obs_lat, obs_lon, obs_data, w_lat, w_lon)
        fcst_by_source['wrf'].append(w_data)
        obs_by_source['wrf'].append(w_obs)
 
        # ---- bc / norway / global ----
        for key, xr_ds in [('bc', bc_ds), ('norway', norway_ds), ('global', global_ds)]:
            vcfg = VARIABLES[key]
            data, lat, lon = load_anemoi_leadtime(xr_ds, li, vcfg['name'])
            data = apply_units(data, vcfg)
            grid_by_source.setdefault(key, (lat, lon))
            o = nearest_obs_on_grid(obs_lat, obs_lon, obs_data, lat, lon)
            fcst_by_source[key].append(data)
            obs_by_source[key].append(o)
 
    bc_ds.close(); norway_ds.close(); global_ds.close()
 
    for key in sources:
        lat2d, lon2d = grid_by_source[key]
        write_verif_file(
            OUTPUT_FILES[key], DATE, LEADTIME_HOURS, lat2d, lon2d,
            obs_by_source[key], fcst_by_source[key],
            LONG_NAME, STANDARD_NAME,
        )
 
 
if __name__ == "__main__":
    main()
 
