"""Utilities for Visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cmaps
import matplotlib.colors as mcolors
import numpy as np
import yaml
from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from shapely.ops import unary_union

# Utilities.
products = ["TREF", "REF", "VEL", "SW", "ZDR", "KDP", "RHO", "SQI"]
varnames = ["mdbz", "T2"]
msg = f"Got {{}}, which should be {' / '.join(products + varnames)}."
# Static Utilities.
ProductType = Literal["TREF", "REF", "VEL", "SW", "ZDR", "KDP", "RHO", "SQI",
                      "mdbz", "T2"]


def get_filename(filedict: dict[str, Path | str]) -> Path:
    """Construct a full file path from dictionary components.

    Args:
        filedict (dict[str, Path | str]): Dictionary with path components.

    Returns:
        Path: Full path combining filepath & folder & filename components.

    """
    keys = ("filepath", "folder", "filename")
    if miss := set(keys) - set(filedict.keys()):
        msg = f"Missing required keys: {', '.join(miss)}."
        raise KeyError(msg)
    filepath, folder, filename = (filedict[key] for key in keys)
    if isinstance(filepath, Path):
        return filepath / folder / filename
    if isinstance(filepath, str):
        return Path(filepath) / folder / filename
    msg = f"Got {type(filepath).__name__}, which should be Path / str."
    raise TypeError(msg)


def get_coverage(*, is_2024: bool) -> GeoSeries:
    """Generate a coverage area as the union around sites.

    Args:
        is_2024 (bool):
            - If True, use only sites relevant for 2024 scenarios.
            - If False, include all available sites.

    Returns:
        GeoSeries: A single geometry representing the union around sites.

    """
    points = [
        {"lon": 118.11581597619593, "lat": 35.64568551155871},  # AnDi
        {"lon": 117.55550469214923, "lat": 35.43144884412909},  # TangCun
        {"lon": 118.86289768825819, "lat": 35.78850995651179},  # QingFengLing
        {"lon": 118.08834973678188, "lat": 36.14557106889448},  # TianZhuang
        {"lon": 118.29159990844587, "lat": 35.96978713664454},  # DongLiDian
        {"lon": 117.71480888075074, "lat": 35.94232089723049},  # JingDou
        {"lon": 118.56000000000000, "lat": 35.17630000000000},  # BaHu
        {"lon": 118.12400000000000, "lat": 35.01590000000000},  # ZhaiShan
    ]

    points_45km = points[:3] if is_2024 else points[:6]
    coverage_45km = GeoDataFrame(
        points_45km,
        geometry=points_from_xy([point["lon"] for point in points_45km],
                                [point["lat"] for point in points_45km]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32649")

    coverage_60km = GeoDataFrame(
        points[-2:],
        geometry=points_from_xy([point["lon"] for point in points[-2:]],
                                [point["lat"] for point in points[-2:]]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32649")

    return GeoSeries(
        unary_union((coverage_45km.buffer(45000).unary_union,
                     coverage_60km.buffer(60000).unary_union)),
        crs="EPSG:32649",
    ).to_crs("EPSG:4326")


def get_cmap(product: ProductType) -> mcolors.ListedColormap:
    """Retrieve a predefined radar product colormap.

    Args:
        product (ProductType): Radar product identifier.

    Returns:
        mcolors.ListedColormap: Matplotlib colormap object.

    """
    match product:
        case "TREF" | "REF" | "mdbz":
            # CMA-MESO.
            # http://nmc.cn/publish/area/china/radar.html
            colors = ["#FFFFFF", "#01A0F6", "#00ECEC", "#00D800", "#019000",
                      "#FFFF00", "#E7C000", "#FF9000", "#FF0000", "#D60000",
                      "#C00000", "#FF00F0", "#9600B4", "#AD90F0"]
            # Single-Site Radar (Range: 00-75 dBZ).
            # https://weather.cma.cn/web/channel-103.html
            colors = ["#0000EF", # Additional for 00-05 dBZ.
                      "#419DF1", "#64E7EB", "#6DFA3D", "#00D800", "#019000",
                      "#FFFF00", "#E7C000", "#FF9000", "#FF0000", "#D60000",
                      "#C00000", "#FF00F0", "#9600B4", "#AD90F0"]
            # National Radar Mosaic.
            # https://data.cma.cn/data/online.html?t=4
            colors = ["#419DF1", "#64E7EB", "#6DFA3D", "#00D800", "#019000",
                      "#FFFF00", "#E7C000", "#FF9000", "#FF0000", "#D60000",
                      "#C00000", "#FF00F0", "#9600B4", "#AD90F0"]
        case "VEL":
            # "#66CCFF" -> "#005AA3".
            colors = ["#005AA3", "#00CCFF", "#009999", "#00FF00", "#00C400",
                      "#006600", "#808080", "#FF0000", "#FF3333", "#FF9999",
                      "#FF6600", "#FFCC00", "#FFFF00"]
        case "ZDR":
            colors = ["#434542", "#6B6D6A", "#949693", "#CBCBD0", "#DCF3DE",
                      "#01C21F", "#00EB0B", "#22FD22", "#FFFE19", "#FFE601",
                      "#FFBC02", "#FF9A00", "#FE5D00", "#F50E00", "#BD003A",
                      "#FF00FD", "#FD00FF"]
        case "KDP":
            colors = ["#00FFFF", "#00F4EF", "#04AAAA", "#B6B6B6", "#B6B6B6",
                      "#00C321", "#00EB0B", "#22FE20", "#FFFF16", "#FEE800",
                      "#FCC000", "#FF9A00", "#FE5E00", "#F80C01", "#BD003A",
                      "#FF00FF"]
        case "RHO":
            colors = ["#003CFC", "#00F2EE", "#00BABB", "#008179", "#008A3A",
                      "#00B729", "#00DA05", "#00FE01", "#FDFE38", "#FFF200",
                      "#FFC600", "#FEA600", "#FF7101", "#FD1C03", "#C50001",
                      "#D500AD"]
        case _:
            raise ValueError(msg.format(product))
    return mcolors.ListedColormap(colors)


def get_ticks(product: ProductType) -> np.ndarray:
    """Retrieve predefined tick values.

    Args:
        product (ProductType): Radar product identifier.

    Returns:
        np.ndarray: Formatted tick positions.

    """
    match product:
        case "TREF" | "REF" | "mdbz":
            ticks = np.linspace(5, 75, 15)
        case "VEL":
            ticks = [-24, -23, -20, -15, -10, -5, -1, 1, 5, 10, 15, 20, 23, 24]
        case "ZDR":
            ticks = [-4.0, -3.0, -2.0, -1.0, 0.0, 0.2, 0.5, 0.8, 1.0,
                      1.5,  2.0,  2.5,  3.0, 3.5, 4.0, 5.0, 6.0]
        case "KDP":
            ticks = [-0.80, -0.40, -0.20, -0.10, 0.10, 0.15,  0.22,  0.33, 0.5,
                      0.75,  1.10,  1.70,  2.40, 3.10, 7.00, 20.00, 21.00]
        case "RHO":
            ticks = [0.00, 0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90,
                     0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        case _:
            raise ValueError(msg.format(product))
    if isinstance(ticks, np.ndarray):
        return ticks
    return np.array(ticks)


def get_clabel(product: ProductType) -> str:
    """Retrieve a formatted LaTeX label for colorbars.

    Args:
        product (ProductType): Radar product identifier.

    Returns:
        str: LaTeX-formatted string for colorbar labeling.

    """
    with Path("viztools.yaml").open() as file:
        info = yaml.safe_load(file)
    if product not in info:
        raise ValueError(msg.format(product))
    label, unit = info[product].values()
    if unit is None:
        return f"$\\mathrm{{ {label} }}$"
    return f"$\\mathrm{{ {label} \\ \\left( {unit} \\right) }}$"


def get_viztools(
    product: ProductType,
    *,
    is_diff: bool = False,
) -> tuple[
    mcolors.ListedColormap,
    np.ndarray,
    mcolors.BoundaryNorm | mcolors.Normalize,
]:
    """Retrieve colormap, tick values, and normalization for visualization.

    Args:
        product (ProductType): Radar product identifier.
        is_diff (bool, optional): Whether for difference plotting.
            Defaults to False.

    Returns:
        tuple[mcolors.ListedColormap, np.ndarray,
              mcolors.BoundaryNorm | mcolors.Normalize]:
            - Matplotlib colormap object.
            - Formatted tick positions.
            - Data normalization.

    """
    # Difference Plots.
    if is_diff:
        match product:
            case "mdbz":
                cmap = cmaps.BlWhRe
                ticks = np.linspace(-20, 20, 9)
                norm = mcolors.Normalize(vmin=-20, vmax=20)
            case "T2":
                cmap = cmaps.BlWhRe
                ticks = np.linspace(-5, 5, 11)
                norm = mcolors.Normalize(vmin=-5, vmax=5)
            case _:
                msg = f"Got {product}, which should be {' / '.join(varnames)}"
                raise ValueError(msg)
    # Standard Plots.
    else:
        match product:
            case "TREF" | "REF" | "VEL" | "ZDR" | "KDP" | "RHO" | "mdbz":
                cmap = get_cmap(product)
                ticks = get_ticks(product)
                norm = mcolors.BoundaryNorm(boundaries=ticks, ncolors=cmap.N)
            case "SW":
                cmap = cmaps.WhiteBlueGreenYellowRed
                ticks = np.linspace(0, 5, 11)
                norm = mcolors.Normalize(vmin=0, vmax=5)
            case "SQI":
                cmap = cmaps.WhiteBlueGreenYellowRed
                ticks = np.linspace(0, 1, 11)
                norm = mcolors.Normalize(vmin=0, vmax=1)
            case "T2":
                cmap = cmaps.WhiteBlueGreenYellowRed
                ticks = np.linspace(285, 300, 7)
                norm = mcolors.Normalize(vmin=285, vmax=300)
            case _:
                raise ValueError(msg.format(product))
    return cmap, ticks, norm
