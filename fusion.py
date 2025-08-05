"""Fuse XPAR & SWAN."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from alive_progress import alive_bar
from cinrad.io import SWAN
from netCDF4 import Dataset
from pandas import date_range
from scipy.interpolate import griddata
from shapely.vectorized import contains
from viztools import get_coverage, get_filename
from wrf import getvar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geopandas import GeoSeries

# Date (format: YYYY-MM-DD).
date = "2025-03-02"
# Start & End (format: HH:MM:SS).
start, end = "06:00:00", "06:00:00"
# Domain.
domain = 2
# MAX / S / X.
mode = "MAX"
# Threshold of Reflectivity for mode S.
threshold = 10


def get_targets(domain: int) -> tuple[np.ndarray]:
    """Retrieve longitude and latitude arrays from a WRF geogrid domain file.

    Args:
        domain (int): WRF domain number.

    Returns:
        tuple[np.ndarray]: Tuple containing longitude and latitude values.

    """
    filepath = Path("/public/home/premopr/data/GMODJOBS/GESDHYD/wps/")
    filename = f"geo_em.d{domain:02}.nc"
    ncfile = Dataset(filepath / filename)
    return (getvar(ncfile, "lon").to_numpy(), getvar(ncfile, "lat").to_numpy())


def base_interpolate(
    data: Dataset,
    index: int,
    bar: Iterable,
    targets: tuple[np.ndarray],
    coverage: GeoSeries,
) -> np.ndarray:
    """Interpolates data from radar grid to WRF grid at specified level.

    Args:
        data (Dataset): Source radar dataset (SWAN or Mosaic).
        index (int): Height level index to select from the radar data.
        bar (Iterable): Progress bar.
        targets (tuple[np.ndarray]): Tuple of target coordinate arrays for WRF.
        coverage (GeoSeries): Geographic boundary defining valid data regions.

    Returns:
        np.ndarray: Interpolated reflectivity on WRF grid at specified height.

    """
    lon = data.variables["longitude"][:]
    lat = data.variables["latitude"][:]
    lon, lat = np.meshgrid(lon, lat)
    points = np.column_stack([lon.ravel(), lat.ravel()])

    if hasattr(data, "CR"):
        # SWAN.
        data = data.CR.isel(height=index).to_numpy()
        bar.text(f"| SWAN: {index + 1} / 21")
    else:
        # Mosaic.
        data = data.variables["REF"][index]
        data[np.isnan(data) & contains(coverage.iloc[0], lon, lat)] = 0
        bar.text(f"| Mosaic: {index + 1} / 21")
    bar()
    return griddata(
        points=points,
        values=data.ravel(),
        xi=targets,
        method="nearest",
        fill_value=np.nan,
    ).reshape(next(iter(targets)).shape)


def fuse(
    swan: list | None,
    mosaic: list,
    mode: Literal["MAX", "S", "X"],
    threshold: int,
) -> np.ndarray:
    """Fuse SWAN data with a mosaic array using specified fusion mode.

    Args:
        swan (list | None): List containing SWAN source data.
            None indicates unnecessary SWAN data.
        mosaic (list): List containing Mosaic source data.
        mode (Literal['MAX', 'S', 'X']): Fusion strategy selector.
            'MAX' - Element-wise maximum between swan and mosaic.
            'S' - Replace values below threshold or NaNs with mosaic values.
            'X' - Output mosaic data exclusively.
        threshold (int): Value limit used for replacement in 'S' mode.

    Returns:
        np.ndarray: Fused array generated according to the specified mode.

    """
    miss = -32.5
    swan = np.array(swan) if swan is not None else None
    mosaic = np.array(mosaic)
    match mode:
        case "MAX":
            data = np.fmax(swan, mosaic)
        case "S":
            data = np.where(np.isnan(swan) | (swan < threshold), mosaic, swan)
        case "X":
            data = mosaic
        case _:
            msg = f"Got {mode}, which should be MAX / S / X."
            raise ValueError(msg)
    mask = np.isnan(data)
    # nan -> -999: To avoid RuntimeWarning: invalid value encountered in cast.
    data = np.nan_to_num(np.where((data == miss) | (data < 0), 0, data), -999)
    return np.ma.masked_array(data, mask, fill_value=-999).astype("int32")


def main() -> None:
    """Fuse."""
    os.chdir(Path(__file__).resolve().parent)

    d02 = 2
    is_d02 = domain == d02
    latitude = slice(38.8, 32.5) if is_d02 else slice(37.5, 33.8)
    longitude = slice(114.5, 123) if is_d02 else slice(115.5, 120)

    year_2024 = "2024"
    is_2024 = date[:4] == year_2024
    coverage = get_coverage(is_2024=is_2024)
    targets = get_targets(domain)
    shape = next(iter(targets)).shape
    interpolate = partial(base_interpolate, targets=targets, coverage=coverage)

    interval = tuple(f"{date} {time}" for time in (start, end))
    times = date_range(*interval, freq="6min")
    total = len(times) * 21 * 2 if mode not in ("X") else len(times) * 21
    with alive_bar(total, spinner="dots", bar="brackets") as bar:
        for time in times:
            folder = Path(time.strftime(r"%Y%m%d%H"))
            filename = time.strftime(r"%Y-%m-%d_%H:%M:%S")
            nctime = list(filename)

            swantime = time.replace(minute=time.minute // 6 * 6, second=0)
            swanname = swantime.strftime(r"%Y%m%d%H%M%S")
            filedict = {
                "swan": {
                    "filepath": Path("/data1/premdev/datainput_arc/radar/raw"),
                    "folder": f"{swanname[:6]}/{swanname[:10]}",
                    "filename": f"Z_OTHE_RADAMOSAIC_{swanname}.bin.bz2",
                },
                "mosaic": {
                    "filepath": Path("data/mosaic"),
                    "folder": time.strftime(r"%Y%m%d"),
                    "filename": f"{filename}.nc",
                },
            }

            if mode not in ("X"):
                bar.text("| SWAN: Loading...")
                swan = (
                    SWAN(get_filename(filedict["swan"]), "3DREF")
                    .get_data()
                    .sel(latitude=latitude, longitude=longitude)
                )
                swan = [interpolate(swan, idx, bar) for idx in range(21)]
            else:
                swan = None

            mosaic = Dataset(get_filename(filedict["mosaic"]))
            height = mosaic.variables["height"][:]
            mosaic = [interpolate(mosaic, idx, bar) for idx in range(21)]

            filepath = Path("data/fusion") / mode / folder
            filepath.mkdir(parents=True, exist_ok=True)
            filename = f"{filename}_DBZ_D{domain}.nc"
            with Dataset(filepath / filename, "w") as ncfile:
                ncfile.createDimension("Time", 1)
                ncfile.createDimension("DateStrLen", 19)
                ncfile.createDimension("vertical", 21)
                ncfile.createDimension("south_north", shape[0])
                ncfile.createDimension("west_east", shape[1])

                dims = ("Time", "DateStrLen")
                data = ncfile.createVariable("Times", "S1", dims)
                data[0, :] = nctime

                dims = ("Time", "vertical")
                data = ncfile.createVariable("Level", "i4", dims)
                data[0, :] = height.astype("int32")

                dims = ("Time", "vertical", "south_north", "west_east")
                data = ncfile.createVariable("dz", "i4", dims, fill_value=-999)
                data[0, :, :, :] = fuse(swan, mosaic, mode, threshold)


if __name__ == "__main__":
    main()
