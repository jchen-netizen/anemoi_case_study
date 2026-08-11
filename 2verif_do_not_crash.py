import os
import datetime
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, latlon_coords
from scipy.spatial import cKDTree
import logging
 
# configure simple logging so each step reports progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)
 
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
    'wrf':    {'name': 'T2', 'scale': 1.0, 'offset': -273.15, 'units': '°C'},
    'cx':     {'name': 'T2', 'scale': 1.0, 'offset': -273.15, 'units': '°C'},
    'bc':     {'name': '2t', 'scale': 1.0, 'offset': -273.15, 'units': '°C'},
    'norway': {'name': '2t', 'scale': 1.0, 'offset': -273.15, 'units': '°C'},
    'global': {'name': '2t', 'scale': 1.0, 'offset': -273.15, 'units': '°C'},
}
 
LONG_NAME = "Temperature"
STANDARD_NAME = "air_temperature"
OUTPUT_UNITS = "°C"
 
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
OUTPUT_DIR = "/Users/jchen/Anemoi/2verif/2t"
OUTPUT_FILES = {
    'wrf':    os.path.join(OUTPUT_DIR, "WRF.nc"),
    'bc':     os.path.join(OUTPUT_DIR, "Anemoi_BC.nc"),
    'norway': os.path.join(OUTPUT_DIR, "Anemoi_Norway.nc"),
    'global': os.path.join(OUTPUT_DIR, "Anemoi_Global.nc"),
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
    using a KDTree instead of a full pairwise distance matrix.
 
    Builds the tree once over the (usually much larger) obs grid, then
    queries it with every target point. This is O(n_obs log n_obs) to
    build and O(n_target log n_obs) to query, vs. the old broadcast
    approach's O(n_target * n_obs) memory footprint -- which is what
    was blowing up memory on real-sized WRF/ClimatEx domains.
    """
    logger.debug("nearest_obs_on_grid: obs.shape=%s, tgt.shape=%s",
                 obs_lat.shape, target_lat.shape)
 
    obs_pts = np.column_stack([obs_lat.ravel(), obs_lon.ravel()])
    tgt_pts = np.column_stack([target_lat.ravel(), target_lon.ravel()])
 
    tree = cKDTree(obs_pts)
    _, nearest_idx = tree.query(tgt_pts, k=1)
 
    matched = obs_data.ravel()[nearest_idx]
    return matched.reshape(target_lat.shape)
 
 
def init_epoch_seconds(dt):
    return int((dt - datetime.datetime(1970, 1, 1)).total_seconds())
 
 
def write_verif_file(filepath, init_dt, leadtime_hours, lat2d, lon2d,
                      obs_2d_by_leadtime, fcst_2d_by_leadtime,
                      long_name, standard_name, units):
    """
    Create or append-to a verif-format NetCDF file.
    `location` = flattened native grid of this forecast source
    (assumed static across different init-date runs).
    """
    if not leadtime_hours:
        logger.warning("Skipping %s: no valid leadtimes available for this source", filepath)
        return
 
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
        v_obs.units = units
        v_fcst.units = units
 
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
    logger.info("Wrote %s (time index %s, init=%s, %s leadtimes, %s locations)",
                filepath, t_index, init_dt, n_leadtime, n_loc)
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    logger.info("Starting verif run for init=%s", DATE)
 
    # sources where "one file has all leadtimes" -- open once
    bc_ds = None
    norway_ds = None
    global_ds = None
 
    try:
        logger.info("Opening bc dataset: %s", bc_path(DATE))
        bc_ds = xr.open_dataset(bc_path(DATE), engine='netcdf4')
    except FileNotFoundError:
        logger.warning("[bc] missing init file %s, skipping this source", bc_path(DATE))
 
    try:
        logger.info("Opening norway dataset: %s", norway_path(DATE))
        norway_ds = xr.open_dataset(norway_path(DATE), engine='netcdf4')
    except FileNotFoundError:
        logger.warning("[norway] missing init file %s, skipping this source", norway_path(DATE))
 
    try:
        logger.info("Opening global dataset: %s", global_path(DATE))
        global_ds = xr.open_dataset(global_path(DATE), engine='netcdf4')
    except FileNotFoundError:
        logger.warning("[global] missing init file %s, skipping this source", global_path(DATE))
 
    sources = ['wrf', 'bc', 'norway', 'global']
    fcst_by_source = {k: [] for k in sources}
    obs_by_source = {k: [] for k in sources}
    leadtime_hours_by_source = {k: [] for k in sources}
    grid_by_source = {}  # captured once per source, assumed static grid
 
    for li in LEADTIME_INDICES:
        hours = li * 6
        valid_dt = DATE + datetime.timedelta(hours=hours)
        logger.info("Processing leadtime %s (hours=%s) valid_dt=%s", li, hours, valid_dt)
 
        # ---- obs (cx): one file per valid time ----
        cx_var = VARIABLES['cx']
        try:
            logger.info("Loading obs (cx) for valid_dt: %s", cx_path(valid_dt))
            obs_data, obs_lat, obs_lon = load_wrf_like(cx_path(valid_dt), cx_var['name'])
            logger.info("Loaded obs data shape %s", obs_data.shape)
        except FileNotFoundError:
            logger.warning("[cx] missing obs file for valid_dt=%s, skipping this valid time", valid_dt)
            continue
        obs_data = apply_units(obs_data, cx_var)
 
        # ---- wrf: hourly files, pick the one matching this valid time ----
        wrf_var = VARIABLES['wrf']
        try:
            wrf_fp = wrf_path(DATE, valid_dt)
            logger.info("Loading WRF file: %s", wrf_fp)
            w_data, w_lat, w_lon = load_wrf_like(wrf_fp, wrf_var['name'])
            w_data = apply_units(w_data, wrf_var)
            logger.info("Loaded WRF data shape %s", w_data.shape)
        except FileNotFoundError:
            logger.warning("[wrf] missing file for valid_dt=%s, skipping this valid time", valid_dt)
        else:
            grid_by_source.setdefault('wrf', (w_lat, w_lon))
            logger.info("Resampling obs onto wrf grid")
            w_obs = nearest_obs_on_grid(obs_lat, obs_lon, obs_data, w_lat, w_lon)
            fcst_by_source['wrf'].append(w_data)
            obs_by_source['wrf'].append(w_obs)
            leadtime_hours_by_source['wrf'].append(hours)
 
        # ---- bc / norway / global ----
        for key, xr_ds in [('bc', bc_ds), ('norway', norway_ds), ('global', global_ds)]:
            if xr_ds is None:
                continue
            vcfg = VARIABLES[key]
            try:
                logger.info("Loading %s leadtime %s from dataset", key, li)
                data, lat, lon = load_anemoi_leadtime(xr_ds, li, vcfg['name'])
            except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
                logger.warning("[%s] missing forecast for valid_dt=%s, skipping this valid time (%s)",
                               key, valid_dt, exc)
                continue
            data = apply_units(data, vcfg)
            grid_by_source.setdefault(key, (lat, lon))
            logger.info("Resampling obs onto %s grid", key)
            o = nearest_obs_on_grid(obs_lat, obs_lon, obs_data, lat, lon)
            fcst_by_source[key].append(data)
            obs_by_source[key].append(o)
            leadtime_hours_by_source[key].append(hours)
 
    if bc_ds is not None:
        logger.info("Closing bc dataset")
        bc_ds.close()
    if norway_ds is not None:
        logger.info("Closing norway dataset")
        norway_ds.close()
    if global_ds is not None:
        logger.info("Closing global dataset")
        global_ds.close()
 
    for key in sources:
        if key not in grid_by_source:
            logger.warning("Skipping %s: no usable grid available", OUTPUT_FILES[key])
            continue
        lat2d, lon2d = grid_by_source[key]
        write_verif_file(
            OUTPUT_FILES[key], DATE, leadtime_hours_by_source[key], lat2d, lon2d,
            obs_by_source[key], fcst_by_source[key],
            LONG_NAME, STANDARD_NAME, OUTPUT_UNITS
        )
 
 
if __name__ == "__main__":
    main()
 
