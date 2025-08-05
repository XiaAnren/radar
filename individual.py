"""Plot: StandardData & PhasedArrayData."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cinrad.io import PhasedArrayData, StandardData, read_auto
from matplotlib import animation
from viztools import get_clabel, get_viztools

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.collections import QuadMesh
    from matplotlib.text import Text

# GIF / PNG
mode = "GIF"
# Radar site name (AD / TC / QFL / TZ / DLD / JD / BH / ZS).
site = "ZS"
# Time of Radar Data (format: YYYY-MM-DD HH:MM:SS).
time = "2024-10-17 10:00:00"
# Index of elevation angle (Range: 0 -> 42).
tilt = 0
# List of Products.
# All: "TREF", "REF", "VEL", "SW", "ZDR", "KDP", "RHO", "SQI"
products = ["REF"]
# Whether to Mask (Only for TREF).
mask = False

# Preprocess.
time = datetime.strptime(time, r"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
tilt = 0 if mode == "GIF" else tilt
products = ["TREF"] if mask else products
# Save.
folder = time.strftime(r"%Y-%m-%d_%H:%M:%S")
savepath = Path("images/individual/bzip2") / site / folder


def get_file(site: str, time: datetime) -> StandardData | PhasedArrayData:
    """Get radar file.

    Args:
        site (str): Radar site name (AD / TC / QFL / TZ / DLD / JD).
        time (datetime): Time of Radar Data (format: YYYY-MM-DD HH:MM:SS).

    Returns:
        StandardData | PhasedArrayData: Radar file.

    """
    filepath = Path("data/bzip2")
    folder = f"{site}/{time.strftime(r'%Y%m%d')}"
    basepath = filepath / folder
    modes = ("HAXPT0164", "__", "AXPT0364")
    pattern = re.compile(
        r"Z_RADR_I_(?:ZSD|ZLY)\d{2}_(\d{14})_"
        r"O_DOR_(?:HAXPT0164|AXPT0364)?_CRA_FMT.bin.bz2",
    )

    start = time - timedelta(seconds=15)
    end = time + timedelta(seconds=15)
    for file in chain(*(basepath.glob(f"*{mode}*") for mode in modes)):
        if match := pattern.fullmatch(file.name):
            time = match.group(1)
            time = datetime.strptime(time, r"%Y%m%d%H%M%S").replace(tzinfo=UTC)
            if start <= time <= end:
                time = time.strftime(r"%Y%m%d%H%M%S")
                break

    sid = {"AD": "ZSD01", "TC":  "ZSD02", "QFL": "ZSD03",
           "TZ": "ZSD04", "DLD": "ZSD05", "JD":  "ZSD06",
           "BH": "ZLY01", "ZS":  "ZLY02"}[site]
    if site in ("AD", "TC", "QFL"):
        mode = "HAXPT0164"
    elif site in ("TZ", "DLD", "JD"):
        mode = ""
    elif site in ("BH", "ZS"):
        mode = "AXPT0364"
    filename = f"Z_RADR_I_{sid}_{time}_O_DOR_{mode}_CRA_FMT.bin.bz2"
    return read_auto(filepath / folder / filename)


def ax2ax(ax: GeoAxes, *, bottom: bool = True, left: bool = True) -> None:
    """GeoAxes normalization.

    Args:
        ax (GeoAxes): Input GeoAxes.
        bottom (bool, optional): Whether bottom ticks is displayed.
            Defaults to True.
        left (bool, optional): Whether left ticks is displayed.
            Defaults to True.

    """
    if bottom:
        bottom = "x"
    if left:
        left = "y"
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
        xlocs=[117.5, 118, 118.5, 119, 119.5],
        ylocs=[34.5, 34.75, 35, 35.25, 35.5, 35.75, 36, 36.25],
    )
    ax.spines["geo"].set_linewidth(0.6)
    ax.tick_params(labelsize=16)


def update(
    frame: int,
    file: StandardData | PhasedArrayData,
    pcm: QuadMesh,
    text: Text,
    aux: tuple[str, int, Any],
) -> None:
    """Update the frames.

    Args:
        frame (int): Frames to be updated.
        file (StandardData | PhasedArrayData): Input Radar Data.
        pcm (QuadMesh): QuadMesh to be updated.
        text (Text): Text to be updated.
        aux (tuple[str, int, Any]): Auxiliary information.

    """
    index = frame
    product, drange, bar = aux
    elevation = file.el[index]
    data = file.get_data(index, drange, product)
    if mask:
        masks = np.isnan(file.get_data(index, drange, "REF").REF)
        cdata = np.where(masks, getattr(data, product), np.nan)
    else:
        cdata = getattr(data, product)
    pcm.set_array(cdata)
    text.set_text(f"$\\mathrm{{Elevation \\ Angle: {elevation:.1f} ^\\circ}}$")
    if frame != 0:
        bar.text(f"Product: {product} | Frame: {frame:02} / 20")
        bar()


def main() -> None:
    """Plot."""
    os.chdir(Path(__file__).resolve().parent)

    # Filename  | Type            | Available Products
    # HAXPT0164 | StandardData    | TREF REF VEL SW ZDR PHI KDP RHO SNRH SQI
    # Rt        | PhasedArrayData | TREF REF VEL SW ZDR PHI KDP RHO
    file = get_file(site, time)

    # Radius of data.
    if isinstance(file, StandardData):
        drange = 60
    elif isinstance(file, PhasedArrayData):
        drange = np.round(len(file.data[tilt]["TREF"][0]) * file.reso)

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=file.get_data(0, 60, "TREF").site_longitude,
        central_latitude=file.get_data(0, 60, "TREF").site_latitude,
    )

    fig = plt.figure(figsize=(6, 5))
    with alive_bar(len(products) * 20, spinner="dots", bar="brackets") as bar:
        for product in products:
            text = f"Product: {product}"
            text += " | Frame: 00 / 20" if mode == "GIF" else ""
            bar.text(text)

            ax = fig.subplots(1, 1, subplot_kw={"projection": proj})
            cmap, ticks, norm = get_viztools(product)
            ax2ax(ax)

            data = file.get_data(tilt, drange, product)
            if mask:
                masks = np.isnan(file.get_data(tilt, drange, "REF").REF)
                cdata = np.where(masks, getattr(data, product), np.nan)
            else:
                cdata = getattr(data, product)
            pcm = ax.pcolormesh(
                data.longitude,
                data.latitude,
                cdata,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
            )
            ax.text(
                0.155,
                0.925,
                file.scantime.strftime("%Y-%m-%d\n%H:%M:%S"),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
                bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
            )
            text = ax.text(
                0.975,
                0.9425,
                f"$\\mathrm{{Elevation \\ Angle:{file.el[tilt]:.1f}^\\circ}}$",
                ha="right",
                transform=ax.transAxes,
                fontsize=16,
                bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
            )

            cbar = fig.colorbar(
                pcm,
                orientation="vertical",
                cax=fig.add_axes([0.845, 0.125, 0.02, 0.75]),
                ticks=ticks,
            )
            cbar.set_label(get_clabel(product=product), fontsize=16)
            cbar.ax.tick_params(labelsize=16)

            savepath.mkdir(parents=True, exist_ok=True)
            savename = f"{product}-mask" if mask else product

            if mode == "GIF":
                ani = animation.FuncAnimation(
                    fig=fig,
                    func=partial(
                        update,
                        file=file,
                        pcm=pcm,
                        text=text,
                        aux=(product, drange, bar),
                    ),
                    frames=21,
                    interval=500,
                )
                ani.save(savepath / f"{savename}.gif", writer="pillow")
                fig.clf()
            elif mode == "PNG":
                fig.savefig(savepath / f"{savename}.png", bbox_inches="tight")
                fig.clf()
            else:
                msg = f"Got {mode}, which should be GIF / PNG."
                raise ValueError(msg)


if __name__ == "__main__":
    main()
