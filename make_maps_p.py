"""
make_maps_p.py

Loops over a set of forecast initialization dates and leadtimes, and for each
one produces map plots (filled a variable + contoured geopotential height)
for four forecast sources:
    - Anemoi (BC)
    - Anemoi (Norway)
    - Anemoi (Global)
    - WRF (processed)

and, where a matching observation time exists, ClimatEx is treated as the "ground truth"/observation.
"""

import os
import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pypalettes import create_cmap

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


# ======================================================================
# CONFIG -- edit this block for each run
# ======================================================================

# ---- Init dates to loop over -----------------------------------------
# One datetime.datetime per forecast cycle you want to process.
INIT_DATES = [
    datetime.datetime(2023, 5, 8),
    datetime.datetime(2023, 5, 9),
    datetime.datetime(2023, 5, 10)
]

# ---- Leadtimes to loop over --------------------------------------------
# correspond to valid_hour = leadtime * 6, i.e. 0,6,12,...,84.
LEADTIMES = list(range(0, 15))  # 0 .. 14 inclusive

# WRF's time index is offset by -1 relative to the Anemoi leadtime index
# cuz it doesn't have 0h. leadtime_wrf < 0 is skipped automatically.
WRF_LEADTIME_OFFSET = -1

# ---- Models -----------------------------
RUN_MODELS = {
    "anemoi_bc": True,
    "anemoi_norway": True,
    "anemoi_global": True,
    "wrf": True,
    "climatex_obs": True,  # only plotted when a valid_time match exists
}

# ---- Pressure levels ------------------------------------
# LEVEL is the actual hPa value
# (z_{LEVEL}, u_{LEVEL}, v_{LEVEL}).
LEVEL = 250

# LEVEL_WRF / LEVEL_CX are weird
# wrf: 50 100 250 500 850
# cx: 1000.,  925.,  850.,  700.,  500.,  400.,  300.,  250.,  200.,  150., 100., 50.
LEVEL_WRF = 2
LEVEL_CX = 7

# Geopotential contour range (dam) -- depends on LEVEL, tune per run.
VMIN_LINE = 980
VMAX_LINE = 1040
LINE_STEP = 2

# Wind speed fill range (km/h) -- depends on LEVEL, tune per run.
VMIN_WSPD = 0
VMAX_WSPD = 200
WSPD_UNITS = "km/h"

# ---- ClimatEx time --------------------------------------------------
# ClimatEx has gaps in its time coordinate (not a uniform 6-hourly series) :(
CX_TIMES = [
    "2023-05-08T00:00:00", "2023-05-08T18:00:00",
    "2023-05-09T00:00:00", "2023-05-09T06:00:00",
    "2023-05-09T12:00:00", "2023-05-09T18:00:00",
    "2023-05-10T06:00:00", "2023-05-11T00:00:00",
    "2023-05-11T12:00:00", "2023-05-11T18:00:00",
    "2023-05-12T00:00:00", "2023-05-12T06:00:00",
    "2023-05-12T12:00:00", "2023-05-12T18:00:00",
    "2023-05-13T00:00:00", "2023-05-13T06:00:00",
    "2023-05-13T12:00:00", "2023-05-13T18:00:00",
]
CX_TIMES = [datetime.datetime.fromisoformat(t) for t in CX_TIMES]
CX_TIME_INDEX = {t: i for i, t in enumerate(CX_TIMES)}  # datetime -> index

# ---- File path templates -------------------------------------------------
AI_BC_DIR_TMPL = "/Users/jchen/Anemoi/ignition/bc/{ymd}T00.nc"
AI_NORWAY_DIR_TMPL = "/Users/jchen/Anemoi/ignition/norway/{ymd}T00.nc"
AI_GLOBAL_DIR_TMPL = "/Volumes/jchen/Share_anemoi/results/donnie_creek/global-model/{ymd}T00.nc"
WRF_DIR_TMPL = "/Volumes/jchen/Share_anemoi/WRF-forecasts/WAC00WG-01/wrfout_d02_processed_{ymd6}00.nc"

# ClimatEx
CX_DIR = "/Volumes/Scratch/jchen-scratch/climatex_jessie_20230508-20230513"

# ---- Output -----------------------------------------------------------
OUTPUT_DIR = "/Users/jchen/Anemoi/pressure-contours-maps"

# ---- Map / plot appearance ---------------------------------------------
MAP_EXTENT = [-147, -109, 42, 67]  # [lon_min, lon_max, lat_min, lat_max]
MARKER_LON, MARKER_LAT = -122.1530, 57.6498  # fire
FIGSIZE = (10, 8)
CONTOUR_COLOR = "black"
CONTOUR_LINEWIDTH = 1.0

CMAP_WSPD = create_cmap([
    "#FFFFFF", "#DCDCDC", "#9CD6FC", "#479BEA", "#1165C4", "#1CA2A2",
    "#1EC797", "#24B932", "#9CF89C", "#C3F89A", "#FCE971", "#FEC340",
    "#FFA516", "#FA6100", "#E01D04", "#A50607", "#623E32",
], "continuous")

# ======================================================================
# END CONFIG
# ======================================================================


def calc_wind_speed(u, v):
    """Wind speed (km/h) from u/v components (m/s)."""
    return np.sqrt(u**2 + v**2) * 3.6


def _base_map(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.scatter(MARKER_LON, MARKER_LAT, s=20, color="red", marker="*",
               transform=ccrs.PlateCarree(), zorder=5)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect(1.3)


def plot_plus(lats, lons, line_data, map_data, title, save_path,
              cmap="", units="", vmin=None, vmax=None,
              init_time=None, hour=None, valid_time=None,
              levels=None, color="black", linewidth=1.0, gridded=True):
    """
    Regular-grid version (pcolormesh/contour) -- used for WRF (and ClimatEx,
    which is also on a regular lat/lon grid). Set gridded=False to use the
    scattered/triangulated variant (tripcolor/tricontour) for the Anemoi
    outputs instead.
    """
    cmap_obj = colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=FIGSIZE,
                            subplot_kw={"projection": ccrs.PlateCarree()})

    if gridded:
        sc = ax.pcolormesh(lons, lats, map_data, transform=ccrs.PlateCarree(),
                            cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="auto")
    else:
        sc = ax.tripcolor(lons, lats, map_data, transform=ccrs.PlateCarree(),
                           cmap=cmap_obj, vmin=vmin, vmax=vmax)

    if levels is None:
        levels = np.linspace(np.nanmin(line_data), np.nanmax(line_data), 10)

    if gridded:
        cs = ax.contour(lons, lats, line_data, levels=levels, colors=color,
                         linewidths=linewidth, transform=ccrs.PlateCarree())
    else:
        cs = ax.tricontour(lons, lats, line_data, levels=levels, colors=color,
                            linewidths=linewidth, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")

    _base_map(ax)
    plt.colorbar(sc, label=units, shrink=0.6)
    ax.set_title(title)

    if init_time is not None:
        fig.text(0.02, 0.95, f"Init: {init_time}", fontsize=11, ha="left")
    if hour is not None:
        fig.text(0.6, 0.95, f"Hour: {hour}", fontsize=11, ha="left")
    if valid_time is not None:
        fig.text(0.70, 0.95, f"Valid: {valid_time}", fontsize=11, ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_model(name, ds, leadtime_dim, leadtime_idx, level_selector,
                   z_var, u_var, v_var, lat_var, lon_var, title,
                   init_time_str, hour, valid_time_str, save_path, gridded):
    """
    Generic single-model plot: selects the given time/level, computes
    geopotential (dam) + wind speed (km/h), and saves the figure.
    level_selector: dict for .sel() giving the level (e.g. {"level": ...}
                    or {"air_pressure": ...}), or {} if level is baked
                    into the variable name already (Anemoi z_{level} etc).
    """
    sel = {leadtime_dim: ds[leadtime_dim].values[leadtime_idx]}
    sel.update(level_selector)
    tds = ds.sel(**sel)

    z = tds[z_var].values / 100
    z = z.squeeze()
    u = tds[u_var].values
    v = tds[v_var].values
    wspd = calc_wind_speed(u, v).squeeze()

    lat = tds[lat_var].values
    lon = tds[lon_var].values

    plot_plus(
        lat, lon, z, wspd, title=title, save_path=save_path,
        cmap=CMAP_WSPD, units=WSPD_UNITS, vmin=VMIN_WSPD, vmax=VMAX_WSPD,
        init_time=init_time_str, hour=hour, valid_time=valid_time_str,
        levels=np.arange(VMIN_LINE, VMAX_LINE, LINE_STEP),
        color=CONTOUR_COLOR, linewidth=CONTOUR_LINEWIDTH, gridded=gridded,
    )


def run():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ClimatEx is one fixed file for the whole event -- open once.
    cx = None
    if RUN_MODELS.get("climatex_obs"):
        try:
            cx = xr.open_dataset(CX_DIR, engine="netcdf4")
        except Exception as e:
            print(f"[WARN] Could not open ClimatEx file, disabling obs panels: {e}")
            cx = None

    for init_date in INIT_DATES:
        ymd = init_date.strftime("%Y%m%d")
        ymd6 = init_date.strftime("%y%m%d")

        print(f"\n=== Init date {ymd} ===")

        datasets = {}
        if RUN_MODELS.get("anemoi_bc"):
            path = AI_BC_DIR_TMPL.format(ymd=ymd)
            datasets["anemoi_bc"] = xr.open_dataset(path, engine="netcdf4") if os.path.exists(path) else None
        if RUN_MODELS.get("anemoi_norway"):
            path = AI_NORWAY_DIR_TMPL.format(ymd=ymd)
            datasets["anemoi_norway"] = xr.open_dataset(path, engine="netcdf4") if os.path.exists(path) else None
        if RUN_MODELS.get("anemoi_global"):
            path = AI_GLOBAL_DIR_TMPL.format(ymd=ymd)
            datasets["anemoi_global"] = xr.open_dataset(path, engine="netcdf4") if os.path.exists(path) else None
        if RUN_MODELS.get("wrf"):
            path = WRF_DIR_TMPL.format(ymd6=ymd6)
            datasets["wrf"] = xr.open_dataset(path, engine="netcdf4") if os.path.exists(path) else None

        for name, ds in datasets.items():
            if ds is None:
                print(f"  [WARN] {name}: file not found for init {ymd}, skipping all leadtimes")

        for leadtime in LEADTIMES:
            valid_hours = leadtime * 6
            valid_dt = init_date + datetime.timedelta(hours=valid_hours)
            valid_time_str = valid_dt.strftime("%Y-%m-%d %HZ")
            init_time_str = f"{init_date.strftime('%Y-%m-%d')} 00Z"

            leadtime_dir = Path(OUTPUT_DIR) / ymd / f"lt{leadtime:02d}_valid{valid_dt.strftime('%Y%m%dT%H')}"

            # --- Anemoi BC ---
            if datasets.get("anemoi_bc") is not None and leadtime < datasets["anemoi_bc"].sizes.get("lead_time", 0):
                try:
                    process_model(
                        "anemoi_bc", datasets["anemoi_bc"], "lead_time", leadtime, {},
                        f"z_{LEVEL}", f"u_{LEVEL}", f"v_{LEVEL}", "latitude", "longitude",
                        title=f"Anemoi (BC) {LEVEL} hPa Geopotential Heights (dam) and Wind Speed",
                        init_time_str=init_time_str, hour=valid_hours, valid_time_str=valid_time_str,
                        save_path=leadtime_dir / "anemoi_bc.png", gridded=False,
                    )
                except Exception as e:
                    print(f"  [WARN] anemoi_bc lt{leadtime}: {e}")

            # --- Anemoi Norway ---
            if datasets.get("anemoi_norway") is not None and leadtime < datasets["anemoi_norway"].sizes.get("lead_time", 0):
                try:
                    process_model(
                        "anemoi_norway", datasets["anemoi_norway"], "lead_time", leadtime, {},
                        f"z_{LEVEL}", f"u_{LEVEL}", f"v_{LEVEL}", "latitude", "longitude",
                        title=f"Anemoi (Norway) {LEVEL} hPa Geopotential Heights (dam) and Wind Speed",
                        init_time_str=init_time_str, hour=valid_hours, valid_time_str=valid_time_str,
                        save_path=leadtime_dir / "anemoi_norway.png", gridded=False,
                    )
                except Exception as e:
                    print(f"  [WARN] anemoi_norway lt{leadtime}: {e}")

            # --- Anemoi Global ---
            if datasets.get("anemoi_global") is not None and leadtime < datasets["anemoi_global"].sizes.get("lead_time", 0):
                try:
                    process_model(
                        "anemoi_global", datasets["anemoi_global"], "lead_time", leadtime, {},
                        f"z_{LEVEL}", f"u_{LEVEL}", f"v_{LEVEL}", "latitude", "longitude",
                        title=f"Anemoi (Global) {LEVEL} hPa Geopotential Heights (dam) and Wind Speed",
                        init_time_str=init_time_str, hour=valid_hours, valid_time_str=valid_time_str,
                        save_path=leadtime_dir / "anemoi_global.png", gridded=False,
                    )
                except Exception as e:
                    print(f"  [WARN] anemoi_global lt{leadtime}: {e}")

            # --- WRF ---
            leadtime_wrf = leadtime + WRF_LEADTIME_OFFSET
            if (datasets.get("wrf") is not None and leadtime_wrf >= 0
                    and leadtime_wrf < datasets["wrf"].sizes.get("XTIME", 0)):
                try:
                    process_model(
                        "wrf", datasets["wrf"], "XTIME", leadtime_wrf,
                        {"air_pressure": datasets["wrf"]["air_pressure"].values[LEVEL_WRF]},
                        "geopotential", "u", "v", "XLAT", "XLONG",
                        title=f"WRF Processed {LEVEL} hPa Geopotential Heights (dam) and Wind Speed",
                        init_time_str=init_time_str, hour=valid_hours, valid_time_str=valid_time_str,
                        save_path=leadtime_dir / "wrf.png", gridded=True,
                    )
                except Exception as e:
                    print(f"  [WARN] wrf lt{leadtime}: {e}")

            # --- ClimatEx (obs), only if this valid_time exists in CX_TIMES ---
            if cx is not None:
                cx_idx = CX_TIME_INDEX.get(valid_dt)
                if cx_idx is not None:
                    try:
                        process_model(
                            "climatex_obs", cx, "time", cx_idx,
                            {"level": cx["level"].values[LEVEL_CX]},
                            "geopotential", "u", "v", "latitude", "longitude",
                            title=f"ClimatEx {LEVEL} hPa Geopotential Heights (dam) and Wind Speed",
                            init_time_str=None, hour=None, valid_time_str=valid_time_str,
                            save_path=leadtime_dir / "climatex_obs.png", gridded=True,
                        )
                    except Exception as e:
                        print(f"  [WARN] climatex_obs lt{leadtime}: {e}")

            print(f"  leadtime {leadtime:2d} (valid {valid_time_str}) done -> {leadtime_dir}")

        # close per-init-date datasets
        for ds in datasets.values():
            if ds is not None:
                ds.close()

    if cx is not None:
        cx.close()

    print(f"\nAll done. Output under: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()