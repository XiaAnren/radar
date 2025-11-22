"""Section."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from itertools import chain
from pathlib import Path
from subprocess import PIPE, Popen, run
from sys import executable as python
from time import sleep
from typing import TYPE_CHECKING, Callable, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from alive_progress import alive_it
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.axes import Axes
from metpy.interpolate import cross_section
from netCDF4 import Dataset
from pandas import date_range
from viztools import get_clabel, get_coverage, get_geometa, get_viztools
from wrf import CoordPair
from xarray.backends import NetCDF4DataStore

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geopandas import GeoSeries

# Mosaic / Plot-Mosaic / Plot-Fusion.
name = "Plot-Mosaic"
# Date (format: YYYY-MM-DD).
date = "2025-03-02"
# Frequency (format: "[n]min" for minutes, "[n]H" for hours).
freq = "6min"
# Time of Mosaic (format: YYYY-MM-DD HH:MM:SS).
time = "2025-03-02 06:00:00"
# Start & End (format: HH:MM:SS).
if name == "Mosaic":
    start, end = "06:00:00", "06:00:00"
# Start & End.
if "Plot" in name:
    start, end = CoordPair(lat=36, lon=117.5), CoordPair(lat=35.75, lon=119)

# Preprocess.
time = datetime.strptime(time, r"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

# Path: MeteoInfoLab.
milab = "/public/home/XiaAnRen/software/MeteoInfo/milab.sh"
# Path: StandardData & PhasedArrayData.
folder = date.replace("-", "")
standard = Path("data/bzip2") / folder
phase = Path("/data3/gengruomei/radar/radarUTC-standard") / folder
# Functions.
funcs: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Register a function as an available main entry point.

    Args:
        name (str): Unique identifier for the main function.

    Returns:
        Callable: Decorator that registers function and returns it unchanged.

    """

    def wrapper(func: Callable) -> Callable:
        funcs[name] = func
        return func

    return wrapper


def ax2ax(
    ax: GeoAxes | Axes,
    text: str,
    coverage: GeoSeries,
    *,
    is_2024: bool = True,
) -> None:
    """GeoAxes and Axes normalization.

    Args:
        ax (GeoAxes | Axes): Input GeoAxes or Axes.
        text (str): Annotation text to display in the bottom-right corner.
        coverage (GeoSeries): GeoSeries containing geometries to overlay.
        is_2024 (bool, optional): Whether year of input data is 2024.
            Defaults to True.

    """
    if isinstance(ax, GeoAxes):
        china_map = cfeature.ShapelyFeature(
            Reader(
                "/public/home/XiaAnRen/share/data/ChinaShp202503/City.shp",
            ).geometries(),
            ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            alpha=1,
        )
        ax.add_feature(china_map, linewidth=0.5, zorder=2)
        ax.gridlines(
            draw_labels={"bottom": "x", "left": "y"},
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
            xlocs=[117, 117.5, 118, 118.5, 119, 119.5],
            ylocs=[34.5, 34.75, 35, 35.25, 35.5, 35.75, 36, 36.25, 36.5],
        )
        ax.spines["geo"].set_linewidth(0.6)
        ax.set_extent([117, 119.375, 34.4625, 36.25 if is_2024 else 36.625])
        ax.add_geometries(
            coverage.geometry,
            crs=ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            linestyle="-",
            linewidth=1.5,
        )
    elif isinstance(ax, Axes):
        ax.grid(linestyle=":", linewidth=0.3, color="k")
        ax.set_ylim(0, 20000)
        ax.set_ylabel("$\\mathrm{ Height \\ (m) }$", fontsize=16)
        ax.xaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=16, bottom=False, left=False)
    ax.text(
        0.975,
        0.975,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )


def fusion_section(
    data: Dataset,
    start: CoordPair,
    end: CoordPair,
) -> xr.Dataset:
    """Extract a vertical cross-section from a dataset between two coordinates.

    Args:
        data (Dataset): Input dataset containing fusion reflectivity data.
        start (CoordPair): Starting coordinate pair for the cross-section.
        end (CoordPair): Ending coordinate pair for the cross-section.

    Returns:
        xr.Dataset: Dataset containing the vertical cross-section.

    """
    data = xr.open_dataset(NetCDF4DataStore(data)).squeeze("Time")

    (lat, lon), _ = get_geometa(domain=3)
    dims = {"vertical": "height", "south_north": "lat", "west_east": "lon"}
    coords = {
        "lat": lat.to_numpy().mean(axis=1),
        "lon": lon.to_numpy().mean(axis=0),
        "height": data.Level.to_numpy(),
    }

    data = data.rename(dims).assign_coords(coords)
    data = data.drop_vars(("Times", "Level")).metpy.parse_cf()
    return cross_section(data, (start.lat, start.lon), (end.lat, end.lon))


@register("Plot-Fusion")
def plot_fusion() -> None:
    """Plot."""
    is_2024 = time.year == 2024  # noqa: PLR2004

    filepath = Path("data/fusion/MAX")
    folder = time.strftime(r"%Y%m%d%H")
    filename = f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}_DBZ_D3"

    (lat, lon), _ = get_geometa(domain=3)

    fusion = Dataset(filepath / folder / f"{filename}.nc")
    reflectivity = np.nanmax(fusion.variables["dz"][0], axis=0)
    section = fusion_section(fusion, start, end)

    coverage = get_coverage(is_2024=is_2024)
    proj = ccrs.AzimuthalEquidistant(118.25, 35.625)
    cmap, ticks, norm = get_viztools("REF")
    cmap.set_under("none")

    fig = plt.figure(figsize=(12, 5))
    subfigs = fig.subfigures(1, 2, wspace=-0.175)

    subfigs[0].set_facecolor("none")
    ax = subfigs[0].subplots(1, 1, subplot_kw={"projection": proj})
    ax2ax(ax, "Fusion", coverage, is_2024=is_2024)
    pcm = ax.pcolormesh(
        lon,
        lat,
        reflectivity,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [start.lon, end.lon],
        [start.lat, end.lat],
        color="black",
        linewidth=3,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.1575,
        0.925,
        time.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    subfigs[1].set_facecolor("none")
    ax = subfigs[1].subplots(1, 1)
    ax2ax(ax, "Cross-section", coverage, is_2024=is_2024)
    ax.pcolormesh(
        section["lat"].values,
        section["height"].values,
        section["dz"].values,
        cmap=cmap,
        norm=norm,
    )

    cbar = fig.colorbar(pcm, ax=ax, ticks=ticks, shrink=0.95, aspect=30)
    cbar.set_label(get_clabel("REF"), fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    savepath = Path(f"images/section/{time.strftime(r'%Y-%m-%d')}")
    savepath.mkdir(parents=True, exist_ok=True)

    fig.savefig(savepath / f"{filename}.png", bbox_inches="tight")


def mosaic_section(
    data: Dataset,
    start: CoordPair,
    end: CoordPair,
) -> xr.Dataset:
    """Extract a vertical cross-section from a dataset between two coordinates.

    Args:
        data (Dataset): Input dataset containing mosaic reflectivity data.
        start (CoordPair): Starting coordinate pair for the cross-section.
        end (CoordPair): Ending coordinate pair for the cross-section.

    Returns:
        xr.Dataset: Dataset containing the vertical cross-section.

    """
    data = xr.open_dataset(NetCDF4DataStore(data)).metpy.parse_cf()
    return cross_section(data, (start.lat, start.lon), (end.lat, end.lon))


@register("Plot-Mosaic")
def plot_mosaic() -> None:
    """Plot."""
    is_2024 = time.year == 2024  # noqa: PLR2004

    filepath = Path("data/mosaic")
    folder = f"section-{time.strftime(r'%Y%m%d')}"
    filename = f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}"

    mosaic = Dataset(filepath / folder / f"{filename}.nc")
    reflectivity = np.nanmax(mosaic.variables["REF"][:], axis=0)
    section = mosaic_section(mosaic, start, end)

    coverage = get_coverage(is_2024=is_2024)
    proj = ccrs.AzimuthalEquidistant(118.25, 35.625)
    cmap, ticks, norm = get_viztools("REF")

    fig = plt.figure(figsize=(12, 5))
    subfigs = fig.subfigures(1, 2, wspace=-0.175)

    subfigs[0].set_facecolor("none")
    ax = subfigs[0].subplots(1, 1, subplot_kw={"projection": proj})
    ax2ax(ax, "Mosaic", coverage, is_2024=is_2024)
    pcm = ax.pcolormesh(
        mosaic.variables["longitude"][:],
        mosaic.variables["latitude"][:],
        reflectivity,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [start.lon, end.lon],
        [start.lat, end.lat],
        color="black",
        linewidth=3,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.1575,
        0.925,
        time.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    subfigs[1].set_facecolor("none")
    ax = subfigs[1].subplots(1, 1)
    ax2ax(ax, "Cross-section", coverage, is_2024=is_2024)
    ax.pcolormesh(
        section["latitude"].values,
        section["height"].values,
        section["REF"].values,
        cmap=cmap,
        norm=norm,
    )

    cbar = fig.colorbar(pcm, ax=ax, ticks=ticks, shrink=0.95, aspect=30)
    cbar.set_label(get_clabel("REF"), fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    savepath = Path(f"images/section/{time.strftime(r'%Y-%m-%d')}")
    savepath.mkdir(parents=True, exist_ok=True)

    fig.savefig(savepath / f"{filename}.png", bbox_inches="tight")


def get_filename(
    filepath: Path,
    interval: list[str],
    freq: str = "6min",
    relaxation: int = 15,
) -> list[str]:
    """Get filenames matching specific pattern within extended time range.

    Args:
        filepath (Path): Directory path to search for files.
        interval (list[str]): Time range list containing start and end time
            (format: YYYY-MM-DD HH:MM:SS).
        freq (str, optional): Frequency between files. Defaults to 6min.
        relaxation (int, optional): Time tolerance in seconds to extend range.
            Defaults to 15.

    Returns:
        list[str]: List of filenames matching both filename pattern and
            extended time criteria.

    """
    time_format = r"%Y%m%d%H%M%S"
    start, end = interval
    modes = ("HAXPT0164", "__", "AXPT0364")
    pattern = re.compile(
        r"Z_RADR_I_(?:ZSD|ZLY)\d{2}_(\d{14})_"
        r"O_DOR_(?:HAXPT0164|AXPT0364)?_CRA_FMT.bin.bz2",
    )

    times = []
    for date in date_range(start, end, freq=freq):
        date_start = date.replace(tzinfo=UTC) - timedelta(seconds=relaxation)
        date_end = date.replace(tzinfo=UTC) + timedelta(seconds=relaxation)
        for file in chain(*(filepath.glob(f"*{mode}*") for mode in modes)):
            if match := pattern.fullmatch(file.name):
                time = match.group(1)
                time = datetime.strptime(time, time_format).replace(tzinfo=UTC)
                if date_start <= time <= date_end:
                    times.append(time.strftime(time_format))
                    break
        else:
            msg = f"File not found in {filepath} around {date}, please check."
            raise FileNotFoundError(msg)
    return times


def check_proc(
    process: list[tuple[Popen, str]],
    sites: list[str],
    bar: Iterable,
    mode: Literal["S", "Standardization", "CAPPI"],
) -> None:
    """Monitor subprocesses in real-time and update progress bar dynamically.

    Args:
        process (list[tuple[Popen, str]]): List of subprocess entries.
        sites (list[str]): Dynamic site cache list.
        bar (Iterable): Progress bar.
        mode (Literal["S", "Standardization", "CAPPI"]): Identifier in display.

    """
    mode = "Standardization" if mode == "S" else mode
    while process:
        for entry in process.copy():
            proc, site = entry
            code = proc.poll()
            if code is not None:
                process.remove(entry)
                sites.remove(site)
                bar.text(f"{mode}: {' | '.join(sites)}")
        sleep(0.1)


@register("Mosaic")
def mosaic() -> None:
    """Save."""
    year = next(iter(date.split("-")))
    interval = tuple(f"{date} {time}" for time in (start, end))

    # All Sites.
    extension = [] if year == "2024" else ["TZ", "DLD", "JD"]
    sites = ["AD", "TC", "QFL", *extension, "BH", "ZS"]
    # STD Sites.
    standardization = ["TC"] if year == "2024" else sites[:6]

    times = {}
    for site in sites:
        filepath = phase / site if site in standardization else standard / site
        times[site] = get_filename(filepath, interval, freq=freq)
    length = len(times["AD"])

    # [2024] STD: TC      | CAPPI: 5 sites | Mosaic: 5 sites.
    # [2025] STD: 6 sites | CAPPI: 8 sites | Mosaic: 8 sites.
    bar = alive_it(range(length), spinner="dots", bar="brackets")
    for index in bar:
        # [2024] STD: TC.
        if year == "2024":
            bar.text("Standardization: TC")
            time = str(times["TC"][index])
            cmd = [python, "standardization.py", "-s", "TC", "-t", time]
            run(cmd, shell=False, check=True, capture_output=True)  # noqa: S603
        # [2025] STD: 6 sites.
        elif year == "2025":
            process = []
            bar.text(f"Standardization: {' | '.join(standardization)}")
            for site in standardization:
                time = str(times[site][index])
                cmd = [python, "standardization.py", "-s", site, "-t", time]
                proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)  # noqa: S603
                process.append((proc, site))
            check_proc(process, standardization.copy(), bar=bar, mode="S")

        # [2024] CAPPI: 5 sites.
        # [2025] CAPPI: 8 sites.
        process = []
        bar.text(f"CAPPI: {' | '.join(sites)}")
        for site in sites:
            time = str(times[site][index])
            cmd = [milab, "-b", "milab_section.py", "-s", site, "-t", time]
            proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)  # noqa: S603
            process.append((proc, site))
        check_proc(process, sites.copy(), bar=bar, mode="CAPPI")

        # [2024] Mosaic: 5 sites.
        # [2025] Mosaic: 8 sites.
        bar.text("Mosaic")
        time = [str(times[site][index]) for site in sites]
        time_format = r"%Y%m%d%H%M%S"
        sitename = next(iter(times.values()))[index]
        sitetime = datetime.strptime(sitename, time_format).replace(tzinfo=UTC)
        increment = ((sitetime.second + 10) // 20) * 20 - sitetime.second
        mosaic_time = sitetime + timedelta(seconds=increment)
        mosaic_name = datetime.strftime(mosaic_time, time_format)
        cmd = [python, "mosaic_section.py", "-t", *time, "-n", mosaic_name]
        run(cmd, shell=False, check=True, capture_output=True)  # noqa: S603


def main() -> None:
    """Save."""
    os.chdir(Path(__file__).resolve().parent)

    if name not in funcs:
        msg = f"Got {name}, which should be {' / '.join(funcs)}."
        raise ValueError(msg)

    func = funcs[name]
    func()


if __name__ == "__main__":
    main()
