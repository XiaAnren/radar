"""Plot: Case."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_it
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from metrics import FSSCalculator
from netCDF4 import Dataset
from pandas import date_range
from viztools import get_clabel, get_coverage, get_viztools
from wrf import get_cartopy, getvar, latlon_coords

if TYPE_CHECKING:
    from datetime import datetime

    from cartopy.mpl.geoaxes import GeoAxes
    from geopandas import GeoSeries
    from xarray import DataArray


# Name of variable (mdbz / T2).
varname = "mdbz"
# Name of GMODJOBS.
jobname = "GWGRM"
# Date (format: YYYY-MM-DD).
date = "2025-03-02"
# Start of assimilation.
start_of_assimilation = 3
# Start & End (format: HH:MM:SS).
start, end = "10:00:00", "21:00:00"
# Frequency (format: "[n]min" for minutes, "[n]h" for hours).
freq = "1h"
# Domain.
domain = 2
# SWAN / XPAR / Fusion.
mode = "Fusion"

# Utilities.
calculators = {"10dBZ": FSSCalculator(10, 2), "35dBZ": FSSCalculator(35, 2)}
folder = f"{date.replace('-', '')}{start_of_assimilation:02}"
jobpath = {
    "GESDHYD": {
        "CTRL": "swan_case/T2024101706.rerun/WRF_F",
        "SWAN": "swan_case/2024101706_swan/WRF_F",
        "XPAR": "x_case/2024101706_x_d02-03/WRF_F",
    },
    "GWGRM": {
        "CTRL": folder,
        "SWAN": f"{folder}/swan_case",
        "XPAR": f"{folder}/x_case",
        "Fusion": f"{folder}/fusion_case",
    },
}
# Save.
savepath = Path(f"images/case/{jobname}/{folder}/{mode}/D{domain:02}")


def get_geometa(domain: int) -> tuple[tuple[DataArray], ccrs.Projection]:
    """Get geographic coordinates and map projection from a geographic file.

    Args:
        domain (int): WRF domain number.

    Returns:
        tuple[tuple[DataArray], Projection]:
            Latitude and longitude coordinates.
            Cartopy projection object for map transformation.

    """
    filepath = Path("/public/home/premopr/data/GMODJOBS/GESDHYD/wps")
    filename = f"geo_em.d{domain:02}.nc"
    ncfile = Dataset(filepath / filename)
    return latlon_coords(getvar(ncfile, "ter")), get_cartopy(wrfin=ncfile)


def ax2ax(
    ax: GeoAxes,
    text: str,
    coverage: GeoSeries,
    *,
    bottom: bool = True,
    left: bool = True,
) -> GeoAxes:
    """GeoAxes normalization.

    Args:
        ax (GeoAxes): Input GeoAxes.
        text (str): Annotation text to display in the bottom-right corner.
        coverage (GeoSeries): GeoSeries containing geometries to overlay.
        bottom (bool, optional): Whether bottom ticks is displayed.
            Defaults to True.
        left (bool, optional): Whether left ticks is displayed.
            Defaults to True.

    Returns:
        GeoAxes: Normalized GeoAxes.

    """
    if bottom:
        bottom = "x"
    if left:
        left = "y"
    china_map = cfeature.ShapelyFeature(
        Reader(
            "/public/home/XiaAnRen/share/data/ChinaShp202503/Province.shp",
        ).geometries(),
        ccrs.PlateCarree(),
        edgecolor="black",
        facecolor="none",
        alpha=1,
    )
    ax.add_feature(china_map, linewidth=0.5, zorder=2)
    ax.gridlines(
        draw_labels={"bottom": bottom, "left": left},
        dms=False,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
        linestyle=":",
        linewidth=0.3,
        color="black",
        xlabel_style={"size": 16},
        ylabel_style={"size": 16},
        xformatter=LongitudeFormatter(),
        yformatter=LatitudeFormatter(),
    )
    ax.spines["geo"].set_linewidth(0.6)
    ax.tick_params(labelsize=16)
    ax.add_geometries(
        coverage.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor="black",
        facecolor="none",
        linestyle="-",
        linewidth=1.5,
    )
    ax.text(
        0.985,
        0.9375,
        text,
        ha="right",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )
    return ax


def get_obs(
    time: datetime,
    mode: str,
    domain: int,
    jobname: str,
) -> np.ndarray:
    """Retrieve composite radar reflectivity from observation file.

    Args:
        time (datetime): Time of observation file.
        mode (str): Data mode identifier.
        domain (int): WRF domain number.
        jobname (str): GMODJOBS experiment identifier.

    Returns:
        np.ndarray: Composite radar reflectivity data.

    """
    prefix = "X" if "XPAR" in mode else mode
    filepath = f"/data3/gengruomei/datest/{prefix}_radar_process/outdata"
    folder = f"{jobname}/cappi_mask_out/{time.strftime(r'%Y%m%d%H')}"
    filename = f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}_DBZ_D{domain}.nc"
    ncfile = Dataset(Path(filepath) / folder / filename)
    return np.nanmax(ncfile.variables["dz"][0], axis=0)


def get_data(
    folder: str,
    time: datetime,
    varname: str,
    domain: int,
    jobname: str,
) -> np.ndarray:
    """Retrieve WRF output variable data for specified case and time.

    Args:
        folder (str): Case directory name (relative to base path).
        varname (str): WRF output variable name.
        time (datetime): Time of WRF output file.
        domain (int): WRF domain number.
        jobname (str): GMODJOBS experiment identifier,

    Returns:
        np.ndarray: WRF output variable data.

    """
    threshold = 5
    pattern = f"wrfout_d{domain:02}*{time.strftime(r'%Y-%m-%d_%H:%M:%S')}*"
    filepath = Path(f"/data3/gengruomei/datest/cycles/{jobname}/GFS_WCTRL")
    filename = list((filepath / folder).glob(pattern)).pop().name
    ncfile = Dataset(filepath / folder / filename)
    data = getvar(ncfile, varname, meta=False)
    return np.where(data < threshold, 0, data) if varname == "mdbz" else data


def main() -> None:
    """Plot."""
    os.chdir(Path(__file__).resolve().parent)

    d02 = 2
    is_d02 = domain == d02
    is_2024 = date[:4] == "2024"

    filepath = jobpath[jobname]

    (lat, lon), proj = get_geometa(domain)
    coverage = get_coverage(is_2024=is_2024)
    clabel = get_clabel(varname)

    wspace = 0.05 if is_d02 else -0.2
    hspace = 0 if is_d02 else 0.05

    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    interval = tuple(f"{date} {time}" for time in (start, end))
    times = date_range(*interval, freq=freq)
    for time in alive_it(times, spinner="dots", bar="brackets"):
        # Preprocess: Axes.
        axes = fig.subplots(2, 2, subplot_kw={"projection": proj})
        # Preprocess: Data.
        ctrl = get_data(filepath["CTRL"], time, varname, domain, jobname)
        data = get_data(filepath[mode], time, varname, domain, jobname)
        obs = get_obs(time, mode, domain, jobname)

        # Standard Plots: Preprocess.
        cmap, ticks, norm = get_viztools(varname)
        cmap.set_under("white")
        args = {"cmap": cmap, "norm": norm, "transform": ccrs.PlateCarree()}
        # Standard Plots: CTRL.
        ax = ax2ax(axes[0, 0], "CTRL", coverage, bottom=(varname != "mdbz"))
        pcm = ax.pcolormesh(lon, lat, ctrl, **args)
        ax.text(
            0.125 if is_d02 else 0.15,
            0.9805,
            time.strftime("%Y-%m-%d\n%H:%M:%S"),
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
        )
        # Standard Plots: DA.
        ax = ax2ax(axes[0, 1], mode, coverage, left=False, bottom=False)
        pcm = ax.pcolormesh(lon, lat, data, **args)
        # Standard Plots: OBS & Metrics.
        if varname == "mdbz":
            ax = ax2ax(axes[1, 0], f"{mode}(OBS)", coverage)
            pcm = ax.pcolormesh(lon, lat, obs, **args)
            for ax, out in zip((axes[0, 0], axes[0, 1]), (ctrl, data)):
                title = "\\quad".join(
                    f"FSS_{{{threshold}}}: {calculator(out, obs):.2f}"
                    for threshold, calculator in calculators.items()
                )
                ax.set_title(f"$\\mathrm{{ {title} }}$", fontsize=18)
        else:
            axes[1, 0].remove()
        # Standard Plots: Colorbar.
        cbar = fig.colorbar(
            pcm,
            orientation="vertical",
            cax=fig.add_axes([0.915 if is_d02 else 0.865, 0.515, 0.01, 0.35]),
            ticks=ticks,
        )
        cbar.set_label(clabel, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        # Difference Plot: Preprocess.
        cmap, ticks, norm = get_viztools(varname, is_diff=True)
        args = {"cmap": cmap, "norm": norm, "transform": ccrs.PlateCarree()}
        # Difference Plot: DIFF.
        ax = ax2ax(axes[1, 1], f"{mode}(DIFF)", coverage, left=False)
        pcm = ax.pcolormesh(lon, lat, data - obs, **args)
        # Difference Plot: Colorbar.
        cbar = fig.colorbar(
            pcm,
            orientation="vertical",
            cax=fig.add_axes([0.915 if is_d02 else 0.865, 0.125, 0.01, 0.35]),
            ticks=ticks,
        )
        cbar.set_label(clabel, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        # Save.
        savepath.mkdir(parents=True, exist_ok=True)
        savename = f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}-{varname}"
        fig.savefig(savepath / f"{savename}.png", bbox_inches="tight")
        fig.clf()


if __name__ == "__main__":
    main()
