"""Get Mosaic."""

import argparse
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from alive_progress import alive_it
from cinrad.calc import GridMapper
from netCDF4 import Dataset
from xarray import concat, open_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Namespace containing parsed command-line arguments.

    """
    # AD & TC & QFL & BH & ZS
    times = [
        "20241017100000",
        "20241017095953",
        "20241017095959",
        "20241017100000",
        "20241017100000",
    ]
    # AD & TC & QFL & TZ & DLD & JD & BH & ZS
    times = [
        "20250302060600",
        "20250302060559",
        "20250302060603",
        "20250302060613",
        "20250302060550",
        "20250302060600",
        "20250302060559",
        "20250302060600",
    ]

    name = "20241017100000"

    parser = argparse.ArgumentParser(
        description="Get Mosaic with site and time parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--times",
        nargs="+",
        type=str,
        default=times,
        help=(
            "Radar data time in order of AnDi and TangCun and QingFengLing "
            "[and TianZhuang and DongLiDian and JingDou] "
            "and BaHu and ZhaiShan"
            "(format: YYYYMMDDHHMMSS)."
        ),
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=name,
        help="Raw NC file name (format: YYYYMMDDHHMMSS).",
    )
    return parser.parse_args()


def get_dataset(filename: Path, height: int) -> Dataset:
    """Retrieve a dataset from a netCDF file and configure its attributes.

    Args:
        filename (Path): Path to the input netCDF file.
        height (Path): Target height value for data selection.

    Returns:
        Dataset: Output dataset. The actual returned object is xarray.Dataset.
            The netCDF4 import is required for low-level HDF5 error handling
            despite not being directly used in this function.

    """
    file = open_dataset(filename)
    file = file.assign_attrs(
        {
            "scan_time": datetime.strptime(filename.stem, r"%Y%m%d%H%M%S")
            .replace(tzinfo=UTC)
            .strftime(r"%Y-%m-%d %H:%M:%S"),
        },
    )
    return file.sel(height=height)


def main() -> None:
    """Get data."""
    os.chdir(Path(__file__).resolve().parent)

    args = parse_args()

    times = args.times
    name = args.name

    filepath = Path("data/cappi")
    savepath = Path("data/mosaic")

    extension = [] if name[:4] == "2024" else ["TZ", "DLD", "JD"]
    sites = ["AD", "TC", "QFL", *extension, "BH", "ZS"]
    folders = [f"section-{time[:8]}" for time in times]

    heights = np.linspace(0, 19000, 191)

    with warnings.catch_warnings():
        # RuntimeWarning: All-NaN axis encountered.
        # [cinrad.calc.py:477] r_data_max = np.nanmax(r_data, axis=0)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        data = concat(
            [
                GridMapper(
                    [
                        get_dataset(
                            filename=(filepath / folder / site / f"{time}.nc"),
                            height=height,
                        )
                        for folder, site, time in zip(folders, sites, times)
                    ],
                )(step=0.0045)
                for height in alive_it(heights, spinner="dots", bar="brackets")
            ],
            dim="height",
        ).assign_coords(height=heights.astype("int32"))

    time = datetime.strptime(name, r"%Y%m%d%H%M%S").replace(tzinfo=UTC)
    folder = f"section-{time.strftime(r'%Y%m%d')}"
    savename = time.strftime(r"%Y-%m-%d_%H:%M:%S")

    (savepath / folder).mkdir(parents=True, exist_ok=True)

    data.to_netcdf(savepath / folder / f"{savename}.nc")


if __name__ == "__main__":
    main()
