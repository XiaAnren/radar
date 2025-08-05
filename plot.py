"""Plot: Fusion."""

import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cinrad.io import SWAN
from geopandas import GeoSeries
from matplotlib import animation
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
from netCDF4 import Dataset
from viztools import get_clabel, get_coverage, get_filename, get_viztools
from wrf import get_cartopy, getvar, latlon_coords
from xarray import DataArray

# GIF / PNG.
mode = "GIF"
# Time of Mosaic (format: YYYY-MM-DD HH:MM:SS).
time = "2025-03-02 06:00:00"
# Domain.
domain = 2
# MAX / S / X.
folder = "MAX"
# Index of Height (unit: m).
# 00 -> 06: 500,  1000,  1500,  2000,  2500,  3000,  3500.
# 07 -> 13: 4000, 4500,  5000,  5500,  6000,  7000,  8000.
# 14 -> 20: 9000, 10000, 12000, 14000, 15500, 17000, 19000.
index = 1

# Preprocess.
index = 0 if mode == "GIF" else index
time = datetime.strptime(time, r"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def get_geometa(domain: int) -> tuple[tuple[DataArray], ccrs.Projection]:
    """Get geographic coordinates and map projection from a wrfout file.

    Args:
        domain (int): WRF domain number.

    Returns:
        tuple[tuple[DataArray], Projection]:
            Latitude and longitude coordinates.
            Cartopy projection object for map transformation.

    """
    filepath = Path("/public/home/premopr/data/GMODJOBS/GESDHYD/wps/")
    filename = f"geo_em.d{domain:02}.nc"
    ncfile = Dataset(filepath / filename)
    return latlon_coords(getvar(ncfile, "ter")), get_cartopy(wrfin=ncfile)


def ax2ax(
    ax: GeoAxes,
    text: str,
    coverage: GeoSeries,
    *,
    is_d02: bool,
    left: bool = True,
) -> GeoAxes:
    """GeoAxes normalization.

    Args:
        ax (GeoAxes): Input GeoAxes.
        text (str): Annotation text to display in the bottom-right corner.
        coverage (GeoSeries): GeoSeries containing geometries to overlay.
        is_d02 (bool): Whether domain is D02.
        left (bool, optional): Whether left ticks is displayed.
            Defaults to True.

    Returns:
        (GeoAxes): Normalized GeoAxes.

    """
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
        draw_labels={"bottom": "x", "left": left},
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
    extent_d02 = [114.75, 122.85, 33.35, 38.75]
    extent_d03 = [115.905, 119.605, 34.4625, 37.325]
    ax.set_extent(extent_d02 if is_d02 else extent_d03)
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
    return ax


def update(
    frame: int,
    file: tuple[SWAN, Dataset, Dataset],
    pcm: dict[str, QuadMesh],
    text: Text,
) -> None:
    """Update visualization elements for the current animation frame.

    Args:
        frame (int): Current frame index to update.
        file (tuple[SWAN, Dataset, Dataset]): Input SWAN & Mosaic & Fusion.
        pcm (dict[str, QuadMesh]): QuadMesh objects for each visualization.
        text (Text): Matplotlib Text object for height annotation.

    """
    index = frame
    swan, mosaic, fusion = file
    height = mosaic.variables["height"][index]
    pcm["mosaic"].set_array(mosaic.variables["REF"][index])
    pcm["swan"].set_array(swan.CR.isel(height=index))
    pcm["fusion"].set_array(fusion.variables["dz"][0, index])
    text.set_text(f"$ \\mathrm{{ Height: {height:g} \\ m }}$")


def main() -> None:
    """Plot."""
    os.chdir(Path(__file__).resolve().parent)

    year_2024 = 2024
    is_2024 = time.year == year_2024

    d02 = 2
    is_d02 = domain == d02
    latitude = slice(38.8, 32.5) if is_d02 else slice(37.5, 33.8)
    longitude = slice(114.5, 123.5) if is_d02 else slice(115.5, 120)

    daystime = time.strftime(r"%Y%m%d")
    hourtime = time.strftime(r"%Y%m%d%H")
    swanname = time.strftime(r"%Y%m%d%H%M%S")
    filename = time.strftime(r"%Y-%m-%d_%H:%M:%S")

    filedict = {
        "swan": {
            "filepath": Path("/data1/premdev/datainput_arc/radar/raw"),
            "folder": f"{time.strftime(r'%Y%m')}/{hourtime}",
            "filename": f"Z_OTHE_RADAMOSAIC_{swanname}.bin.bz2",
        },
        "mosaic": {
            "filepath": Path("data/mosaic"),
            "folder": daystime,
            "filename": f"{filename}.nc",
        },
        "fusion": {
            "filepath": Path("data/fusion"),
            "folder": f"{folder}/{hourtime}",
            "filename": f"{filename}_DBZ_D{domain}.nc",
        },
    }

    swan = (
        SWAN(get_filename(filedict["swan"]), "3DREF")
        .get_data()
        .sel(latitude=latitude, longitude=longitude)
    )
    mosaic = Dataset(get_filename(filedict["mosaic"]))
    fusion = Dataset(get_filename(filedict["fusion"]))

    coverage = get_coverage(is_2024=is_2024)
    height = mosaic.variables["height"][index]

    (lat, lon), proj = get_geometa(domain)
    cmap, ticks, norm = get_viztools("REF")
    pcm = {}

    fig = plt.figure(figsize=(15, 4) if is_d02 else (12, 3.75))
    fig.subplots_adjust(left=0.075, right=0.895, wspace=0.1)
    axes = fig.subplots(1, 3, subplot_kw={"projection": proj})

    ax = ax2ax(axes[0], "Mosaic", coverage, is_d02=is_d02)
    pcm["mosaic"] = ax.pcolormesh(
        mosaic.variables["longitude"][:],
        mosaic.variables["latitude"][:],
        mosaic.variables["REF"][index],
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.15 if is_d02 else 0.185,
        0.905,
        time.strftime("%Y-%m-%d\n%H:%M:%S"),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )
    text = ax.text(
        0.985,
        0.9275,
        f"$ \\mathrm{{ Height: {height:g} \\ m }}$",
        ha="right",
        transform=ax.transAxes,
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
    )

    ax = ax2ax(axes[1], "SWAN", coverage, is_d02=is_d02, left=False)
    pcm["swan"] = ax.pcolormesh(
        swan.longitude,
        swan.latitude,
        swan.CR.isel(height=index),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    ax = ax2ax(axes[2], "Fusion", coverage, is_d02=is_d02, left=False)
    pcm["fusion"] = ax.pcolormesh(
        lon,
        lat,
        fusion.variables["dz"][0, index],
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    cbar = fig.colorbar(
        next(iter(pcm.values())),
        orientation="vertical",
        cax=fig.add_axes([0.915, 0.15, 0.01, 0.7]),
        ticks=ticks,
    )
    cbar.set_label(get_clabel("REF"), fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    savepath = Path("images/fusion") / folder / daystime
    savepath.mkdir(parents=True, exist_ok=True)
    if mode == "GIF":
        ani = animation.FuncAnimation(
            fig=fig,
            func=partial(
                update,
                file=(swan, mosaic, fusion),
                pcm=pcm,
                text=text,
            ),
            frames=21,
            interval=500,
        )
        ani.save(savepath / f"D{domain:02}_{filename}.gif", writer="pillow")
    elif mode == "PNG":
        fig.savefig(savepath / f"D{domain:02}_{filename}_{height:04g}.png")
    else:
        msg = f"Got {mode}, which should be GIF / PNG."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
