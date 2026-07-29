#!/usr/bin/env python3
import argparse
import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, latlon_coords


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create temperature at 2m comparison maps."
    )
    parser.add_argument(
        "--start-date",
        default="2023-06-05",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2023-06-10",
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--save-dir",
        default="/Users/jchen/Desktop/ignition_plots",
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--ai-bc-root",
        default="/Users/jchen/Anemoi/ignition/bc",
        help="Directory containing Anemoi BC NetCDF files.",
    )
    parser.add_argument(
        "--ai-norway-root",
        default="/Users/jchen/Anemoi/ignition/norway",
        help="Directory containing Anemoi Norway NetCDF files.",
    )
    parser.add_argument(
        "--ai-global-root",
        default="/Volumes/jchen/Share_anemoi/results/donnie_creek/global-model",
        help="Directory containing Anemoi global-model NetCDF files.",
    )
    parser.add_argument(
        "--wrf-root",
        default="/Volumes/jchen/Share_Forecasts/WAC00WG-01",
        help="Directory containing WRF forecast output folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    date_start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    date_end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    current_date = date_start
    while current_date <= date_end:
        date = current_date

        ai_bc_dir = f"{args.ai_bc_root}/{date.strftime('%Y%m%d')}T00.nc"
        ai_norway_dir = f"{args.ai_norway_root}/{date.strftime('%Y%m%d')}T00.nc"
        ai_global_dir = f"{args.ai_global_root}/{date.strftime('%Y%m%d')}T00.nc"

        ai_bc = xr.open_dataset(ai_bc_dir, engine="netcdf4")
        ai_norway = xr.open_dataset(ai_norway_dir, engine="netcdf4")
        ai_global = xr.open_dataset(ai_global_dir, engine="netcdf4")

        ai_norway_ds = Dataset(ai_norway_dir)
        ai_bc_ds = Dataset(ai_bc_dir)
        ai_global_ds = Dataset(ai_global_dir)

        for valid_hours in range(6, 85, 6):
            leadtime = valid_hours // 6
            valid_dt = date + datetime.timedelta(hours=valid_hours)
            valid_time = valid_dt.strftime("%Y-%m-%d %HZ")
            wrf_dir = (
                f"{args.wrf_root}/{date.strftime('%y%m%d')}00/"
                f"wrfout_d02_{valid_dt.strftime('%Y-%m-%d_%H:00:00')}"
            )

            print(f"Working on {date.strftime('%Y-%m-%d')} {leadtime * 6}")

            try:
                wrf = xr.open_dataset(wrf_dir, engine="netcdf4")
            except Exception as exc:
                print(f"  WARNING: could not open {wrf_dir} ({exc}) — skipping this leadtime.")
                continue

            ai_bc_tds = ai_bc.sel(lead_time=ai_bc.lead_time[leadtime])
            ai_bc_var = "2t"
            ai_data = ai_bc_tds[ai_bc_var].values - 273.25
            ai_bc_lat = ai_bc_tds["latitude"].values
            ai_bc_lon = ai_bc_tds["longitude"].values

            ai_norway_tds = ai_norway.sel(lead_time=ai_norway.lead_time[leadtime])
            ai_norway_var = "2t"
            ai_norway_data = ai_norway_tds[ai_norway_var].values - 273.15
            ai_norway_lat = ai_norway_tds["latitude"].values
            ai_norway_lon = ai_norway_tds["longitude"].values

            ai_global_tds = ai_global.sel(lead_time=ai_global.lead_time[leadtime])
            ai_global_var = "2t"
            ai_global_data = ai_global_tds[ai_global_var].values - 273.15
            ai_global_lat = ai_global_tds["latitude"].values
            ai_global_lon = ai_global_tds["longitude"].values

            wrf_var = "T2"
            wrf_ds = Dataset(wrf_dir)
            wrf_data = getvar(wrf_ds, wrf_var).values - 273.15
            wrf_lats, wrf_lons = latlon_coords(getvar(wrf_ds, wrf_var))

            fig, axes = plt.subplots(
                2,
                2,
                gridspec_kw={"wspace": 0, "hspace": 0},
                figsize=(15, 13),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )

            datasets = [
                (ai_bc_lat, ai_bc_lon, ai_data, "Anemoi (BC)"),
                (ai_norway_lat, ai_norway_lon, ai_norway_data, "Anemoi (Norway)"),
                (ai_global_lat, ai_global_lon, ai_global_data, "Anemoi (Global)"),
                (wrf_lats, wrf_lons, wrf_data, "WRF"),
            ]

            axes_flat = axes.flatten()

            for idx, (ax, (lat, lon, data, title)) in enumerate(zip(axes_flat, datasets)):
                row, col = idx // 2, idx % 2

                sc = ax.scatter(
                    lon,
                    lat,
                    c=data,
                    s=0.5,
                    transform=ccrs.PlateCarree(),
                    cmap="RdYlBu_r",
                    vmin=0,
                    vmax=35,
                )
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.add_feature(cfeature.STATES, linewidth=0.3)

                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 5))

                gl.top_labels = row == 0
                gl.bottom_labels = row == 1
                gl.left_labels = col == 0
                gl.right_labels = col == 1

                ax.scatter(
                    -122.1530,
                    57.6498,
                    s=20,
                    color="red",
                    marker="*",
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.set_extent([-108, -148, 43, 67], crs=ccrs.PlateCarree())
                ax.set_aspect("auto")

                if row == 0:
                    ax.text(
                        0.5,
                        1.08,
                        title,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
                else:
                    ax.text(
                        0.5,
                        -0.08,
                        title,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                    )

            fig.text(0.10, 0.95, f"Init: {date.strftime('%Y-%m-%d')} 00Z", fontsize=14, ha="left")
            fig.text(0.70, 0.95, f"Hour: {leadtime * 6}", fontsize=14, ha="left")
            fig.text(0.78, 0.95, f"Valid: {valid_time}", fontsize=14, ha="left")
            fig.suptitle("Temperature at 2 Meters", fontsize=16, y=0.99)

            cbar_ax = fig.add_axes([0.26, 0.035, 0.5, 0.02])
            cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label="kPa")
            cb.set_label("kPa", fontsize=12)
            cb.ax.tick_params(labelsize=10)

            out_path = (
                f"{save_dir}/temp2_{date.strftime('%Y%m%d')}"
                f"_leadtime{valid_hours:03d}h.png"
            )
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            wrf.close()

        ai_bc.close()
        ai_norway.close()
        ai_global.close()

        current_date += datetime.timedelta(days=1)

    print("All done!")


if __name__ == "__main__":
    main()
