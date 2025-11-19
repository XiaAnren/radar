"""Plot: GPM."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from geopandas import GeoSeries
from viztools import get_coverage

# Time (format: YYYY-MM-DD HH:MM:SS).
time = "2024-10-17 12:00:00"

# Preprocess.
time = datetime.strptime(time, r"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def ax2ax(ax: GeoAxes, text: str, coverage: GeoSeries) -> None:
    """GeoAxes normalization.

    Args:
        ax (GeoAxes): Input GeoAxes.
        text (str): Annotation text to display in the bottom-right corner.
        coverage (GeoSeries): GeoSeries containing geometries to overlay.

    """
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
    )
    ax.spines["geo"].set_linewidth(0.6)
    ax.set_extent([114.75, 122.85, 38.75, 33.35])
    ax.add_geometries(
        coverage.geometry,
        crs=ccrs.PlateCarree(),
        edgecolor="black",
        facecolor="none",
        linestyle="-",
        linewidth=1.5,
    )
    ax.tick_params(labelsize=16)
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


def get_filename(time: datetime) -> str:
    """Generate a GPM IMERG HDF5 filename based on the specified time.

    Args:
        time (datetime): The reference datetime for which to generate filename.

    Returns:
        str: Formatted GPM IMERG HDF5 filename.

    """
    template = "3B-HHR.MS.MRG.3IMERG.{date}-S{start}-E{end}.{suffix}.V07B.HDF5"
    date = time.strftime(r"%Y%m%d")
    start = time.strftime(r"%H%M%S")
    end = (time + timedelta(minutes=29, seconds=59)).strftime(r"%H%M%S")
    suffix = str(time.hour * 60 + time.minute).rjust(4, "0")
    return template.format(date=date, start=start, end=end, suffix=suffix)


def main() -> None:
    """Test."""
    os.chdir(Path(__file__).resolve().parent)

    is_2024 = time.year == 2024  # noqa: PLR2004

    filepath = Path("data/GPM")
    with h5py.File(filepath / get_filename(time)) as file:
        lons = file["/Grid/lon"][:]
        lats = file["/Grid/lat"][:]
        data = file["/Grid/precipitation"][0]

    coverage = get_coverage(is_2024=is_2024)
    proj = ccrs.AzimuthalEquidistant(118.25, 35.625)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.subplots(1, 1, subplot_kw={"projection": proj})
    ax2ax(ax, "GPM", coverage)
    pcm = ax.pcolormesh(
        lons[2800:3200],
        lats[1200:1300],
        data[2800:3200, 1200:1300].T,
        cmap=cmaps.WhiteBlueGreenYellowRed,
        norm=mcolors.Normalize(vmin=0, vmax=10),
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.1575,
        0.975,
        time.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    label = "$\\mathrm{ Precipitation \\ \\left( mm \\cdot hr^{-1} \\right) }$"
    cbar = fig.colorbar(
        pcm,
        ax=ax,
        ticks=np.linspace(0, 10, 6),
        pad=0.015,
        shrink=0.75,
        aspect=25,
    )
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    savepath = Path("images/GPM")
    savepath.mkdir(parents=True, exist_ok=True)
    savename = f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.png"
    fig.savefig(savepath / savename, bbox_inches="tight")


if __name__ == "__main__":
    main()
