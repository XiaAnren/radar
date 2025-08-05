"""Plot: Sequence."""

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_it
from matplotlib.lines import Line2D
from metrics import FSSCalculator
from netCDF4 import Dataset
from pandas import Timestamp, date_range
from wrf import getvar

# Name of variable (mdbz).
varname = "mdbz"
# Name of GMODJOBS.
jobname = "GWGRM"
# Date (format: YYYY-MM-DD).
date = "2025-03-02"
# Start of assimilation.
start_of_assimilation = 3
# End of assimilation.
end_of_assimilation = 9
# Start & End (format: HH:MM:SS).
start, end = "04:00:00", "21:00:00"
# Frequency (format: "[n]min" for minutes, "[n]h" for hours).
freq = "1h"
# Domain.
domain = 2
# SWAN / XPAR / Fusion.
mode = "Fusion"

# Utilities.
linestyles = {"CTRL": "-", mode: "--"}
colors = {10: "red", 15: "blue", 30: "green"}
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


def get_obs(
    time: Timestamp,
    mode: str,
    domain: int,
    jobname: str,
) -> np.ndarray:
    """Retrieve composite radar reflectivity from observation file.

    Args:
        time (Timestamp): Time of observation file.
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
    time: Timestamp,
    varname: str,
    domain: int,
    jobname: str,
) -> np.ndarray:
    """Retrieve WRF output variable data for specified case and time.

    Args:
        folder (str): Case directory name (relative to base path).
        varname (str): WRF output variable name.
        time (Timestamp): Time of WRF output file.
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

    units = {"mdbz": "dBZ"}

    if varname not in units:
        msg = f"Got {varname}, which should be {' / '.join(units.keys())}."
        raise ValueError(msg)

    unit = units[varname]
    thresholds = colors.keys()
    filepath = jobpath[jobname]

    metrics, calculators, labels = {}, {}, {}
    for threshold in thresholds:
        metrics[threshold] = {"CTRL": [], mode: []}
        calculators[threshold] = FSSCalculator(threshold, 2)
        labels[threshold] = f"$\\mathrm{{ {threshold} \\ {unit} }}$"

    fig = plt.figure(figsize=(6, 5))
    ax = fig.subplots(1, 1)

    interval = tuple(f"{date} {time}" for time in (start, end))
    times = date_range(*interval, freq=freq)
    for time in alive_it(times, spinner="dots", bar="brackets"):
        ctrl = get_data(filepath["CTRL"], time, varname, domain, jobname)
        data = get_data(filepath[mode], time, varname, domain, jobname)
        obs = get_obs(time, mode, domain, jobname)

        for threshold, calculator in calculators.items():
            metrics[threshold]["CTRL"].append(calculator(ctrl, obs))
            metrics[threshold][mode].append(calculator(data, obs))

    for threshold, color in colors.items():
        ax.plot(times, metrics[threshold]["CTRL"], linestyle="-", color=color)
        ax.plot(times, metrics[threshold][mode], linestyle="--", color=color)
    time = Timestamp(f"{date} {end_of_assimilation:02}")
    ax.axvline(time, color="black", linestyle="--", alpha=0.5)

    args = {"fontsize": 16, "framealpha": 1, "edgecolor": "1"}
    handles = [
        Line2D([], [], color="black", linestyle=linestyle, label=label)
        for label, linestyle in linestyles.items()
    ]
    legend = ax.legend(handles=handles, loc=(0.025, 0.005), **args)
    ax.add_artist(legend)
    handles = [
        Line2D([], [], color=colors[threshold], label=labels[threshold])
        for threshold in thresholds
    ]
    legend = ax.legend(handles=handles, loc=(0.325, 0.0025), **args)
    legend.set_zorder(1.5)

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0, 1)
    ax.set_ylabel("$\\mathrm{{ Fraction \\ Skill \\ Score }}$", fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.grid(linestyle=":", linewidth=0.3, color="black")
    ax.minorticks_on()
    ax.tick_params(labelsize=16)

    savepath.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath / f"sequence-{varname}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
