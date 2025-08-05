"""Plot: SWAN & Mosaic."""

import os
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cinrad.io import SWAN
from geopandas import GeoSeries
from matplotlib import animation
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
from netCDF4 import Dataset
from shapely.vectorized import contains
from viztools import get_clabel, get_coverage, get_filename, get_viztools

# GIF / PNG.
mode = "GIF"
# Index of Height (unit: m).
# 00 -> 06: 500,  1000,  1500,  2000,  2500,  3000,  3500.
# 07 -> 13: 4000, 4500,  5000,  5500,  6000,  7000,  8000.
# 14 -> 20: 9000, 10000, 12000, 14000, 15500, 17000, 19000.
index = 1
# Time of Mosaic (format: YYYY-MM-DD HH:MM:SS).
time = "2025-03-02 06:00:00"

# Preprocess.
index = 0 if mode == "GIF" else index
time = datetime.strptime(time, r"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def ax2ax(
    ax: GeoAxes,
    text: str,
    coverage: GeoSeries,
    *,
    is_2024: bool = True,
    bottom: bool = True,
) -> None:
    """GeoAxes normalization.

    Args:
        ax (GeoAxes): Input GeoAxes.
        text (str): Annotation text to display in the bottom-right corner.
        coverage (GeoSeries): GeoSeries containing geometries to overlay.
        is_2024 (bool, optional): Whether year of input data is 2024.
            Defaults to True.
        bottom (bool, optional): Whether bottom ticks is displayed.
            Defaults to True.

    """
    if bottom:
        bottom = "x"
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
        draw_labels={"bottom": bottom, "left": "y"},
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
    ax.set_extent([117, 119.375, 34.425, 36.25 if is_2024 else 36.625])
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
        0.975,
        0.025,
        text,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )


def update(
    frame: int,
    file: tuple[SWAN, Dataset],
    pcm: tuple[QuadMesh],
    text: Text,
    mask: np.ndarray,
) -> None:
    """Update visualization elements for the current animation frame.

    Args:
        frame (int): Current frame index to update.
        file (tuple[SWAN, Dataset]): Input SWAN & Mosaic.
        pcm (tuple[QuadMesh]): QuadMesh objects for each visualization.
        text (Text): Matplotlib Text object for height annotation.
        mask (np.ndarray): 2D boolean array for spatial filtering.

    """
    index = frame
    swan, mosaic = file
    height = mosaic.variables["height"][index]
    pcm[0].set_array(mosaic.variables["REF"][index])
    pcm[1].set_array(np.where(mask, swan.CR.isel(height=index), 0))
    text.set_text(f"$ \\mathrm{{ Height: {height:g} \\ m }}$")


def main() -> None:
    """Plot."""
    os.chdir(Path(__file__).resolve().parent)

    year_2024 = 2024
    is_2024 = time.year == year_2024

    folder = time.strftime(r"%Y%m%d")
    filename = time.strftime(r"%Y-%m-%d_%H:%M:%S")

    minutes_per_hour = 60
    auxtime = time + timedelta(seconds=20)
    minute = (auxtime.minute + 5) // 6 * 6
    swantime = (
        auxtime.replace(minute=minute, second=0)
        if minute != minutes_per_hour
        else auxtime.replace(minute=0, second=0) + timedelta(hours=1)
    )
    swanname = swantime.strftime(r"%Y%m%d%H%M%S")

    filedict = {
        "swan": {
            "filepath": Path("/data1/premdev/datainput_arc/radar/raw"),
            "folder": f"{swanname[:6]}/{swanname[:10]}",
            "filename": f"Z_OTHE_RADAMOSAIC_{swanname}.bin.bz2",
        },
        "mosaic": {
            "filepath": Path("data/mosaic"),
            "folder": folder,
            "filename": f"{filename}.nc",
        },
    }

    latitude = slice(36.5 if is_2024 else 36.8, 34.5)
    swan = (
        SWAN(get_filename(filedict["swan"]), "3DREF")
        .get_data()
        .sel(latitude=latitude, longitude=slice(116.8, 119.5))
    )
    mosaic = Dataset(get_filename(filedict["mosaic"]))

    lon, lat = np.meshgrid(swan.longitude, swan.latitude)
    height = mosaic.variables["height"][index]

    coverage = get_coverage(is_2024=is_2024)
    mask = contains(coverage.iloc[0], lon, lat)

    proj = ccrs.AzimuthalEquidistant(118.25, 35.625)
    cmap, ticks, norm = get_viztools("REF")

    fig = plt.figure(figsize=(7.25, 9.5) if is_2024 else (6.5, 10))
    subfigs = fig.subfigures(2, 1, hspace=-0.2)

    subfigs[0].set_facecolor("none")
    ax = subfigs[0].subplots(1, 1, subplot_kw={"projection": proj})
    ax2ax(ax, "Mosaic", coverage, is_2024=is_2024, bottom=False)
    pcm0 = ax.pcolormesh(
        mosaic.variables["longitude"][:],
        mosaic.variables["latitude"][:],
        mosaic.variables["REF"][index],
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.135 if is_2024 else 0.15,
        0.925,
        time.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )
    text = ax.text(
        0.985,
        0.9425,
        f"$ \\mathrm{{ Height: {height:g} \\ m }}$",
        ha="right",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    subfigs[1].set_facecolor("none")
    ax = subfigs[1].subplots(1, 1, subplot_kw={"projection": proj})
    ax2ax(ax, "SWAN", coverage, is_2024=is_2024)
    pcm = ax.pcolormesh(
        swan.longitude,
        swan.latitude,
        np.where(mask, swan.CR.isel(height=index), 0),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.135 if is_2024 else 0.15,
        0.925,
        swantime.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    cbar = fig.colorbar(
        pcm,
        orientation="vertical",
        cax=fig.add_axes([0.865 if is_2024 else 0.85, 0.125, 0.02, 0.75]),
        ticks=ticks,
    )
    cbar.set_label(get_clabel("REF"), fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    savepath = Path(f"images/swan/{time.strftime(r'%Y-%m-%d')}")
    savepath.mkdir(parents=True, exist_ok=True)
    if mode == "GIF":
        ani = animation.FuncAnimation(
            fig=fig,
            func=partial(
                update,
                file=(swan, mosaic),
                pcm=(pcm0, pcm),
                text=text,
                mask=mask,
            ),
            frames=21,
            interval=500,
        )
        ani.save(savepath / f"{filename}.gif", writer="pillow")
    elif mode == "PNG":
        fig.savefig(savepath / f"{filename}_{height:04g}.png")
    else:
        msg = f"Got {mode}, which should be GIF / PNG."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
