"""Run All Related Scripts to Get Final Mosaic Data."""

import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from itertools import chain
from pathlib import Path
from subprocess import PIPE, Popen, run
from sys import executable as python
from time import sleep
from typing import Literal

from alive_progress import alive_it
from pandas import date_range

# Date (format: YYYY-MM-DD).
date = "2025-03-02"
# Start & End (format: HH:MM:SS).
start, end = "06:00:00", "06:00:00"
# Frequency (format: "[n]min" for minutes, "[n]H" for hours).
freq = "6min"

# Path: MeteoInfoLab.
milab = "/public/home/XiaAnRen/software/MeteoInfo/milab.sh"
# Path: StandardData & PhasedArrayData.
folder = date.replace("-", "")
standard = Path("data/bzip2") / folder
phase = Path("/data3/gengruomei/radar/radarUTC-standard") / folder


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


def check_length(times: dict[str, list[str]]) -> int:
    """Check if all required time lists in the dictionary have equal lengths.

    Args:
        times (dict[str, list[str]]): Dictionary containing site and time.

    Returns:
        int: The common length of all time lists if they are consistent.

    """
    len_2024, len_2025 = 5, 8
    length = len(times)
    if length == len_2024 and not (
        len(times["AD"])
        == len(times["TC"])
        == len(times["QFL"])
        == len(times["BH"])
        == len(times["ZS"])
    ):
        msg = (
            f"Length of AnDi ({len(times['AD'])}) "
            f"& TangCun ({len(times['TC'])}) "
            f"& QingFengLing ({len(times['QFL'])}) "
            f"& BaHu ({len(times['BH'])}) "
            f"& ZhaiShan ({len(times['ZS'])}) do not match, "
            "please check:\n"
            f"AnDi:\n{times['AD']}\n"
            f"TangCun:\n{times['TC']}\n"
            f"QingFengLing:\n{times['QFL']}\n"
            f"BaHu:\n{times['BH']}\n"
            f"ZhaiShan:\n{times['ZS']}"
        )
        raise RuntimeError(msg)
    if length == len_2025 and not (
        len(times["AD"])
        == len(times["TC"])
        == len(times["QFL"])
        == len(times["TZ"])
        == len(times["DLD"])
        == len(times["JD"])
        == len(times["BH"])
        == len(times["ZS"])
    ):
        msg = (
            f"Length of AnDi ({len(times['AD'])}) "
            f"& TangCun ({len(times['TC'])}) "
            f"& QingFengLing ({len(times['QFL'])}) "
            f"& TianZhuang ({len(times['TZ'])}) "
            f"& DongLiDian ({len(times['DLD'])}) "
            f"& JingDou ({len(times['JD'])}) "
            f"& BaHu ({len(times['BH'])}) "
            f"& ZhaiShan ({len(times['ZS'])}) do not match, "
            "please check:\n"
            f"AnDi:\n{times['AD']}\n"
            f"TangCun:\n{times['TC']}\n"
            f"QingFengLing:\n{times['QFL']}\n"
            f"TianZhuang:\n{times['TZ']}\n"
            f"DongLiDian:\n{times['DLD']}\n"
            f"JingDou:\n{times['JD']}\n"
            f"BaHu:\n{times['BH']}\n"
            f"ZhaiShan:\n{times['ZS']}"
        )
        raise RuntimeError(msg)
    return len(next(iter(times.values())))


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


def main() -> None:
    """Get data."""
    os.chdir(Path(__file__).resolve().parent)

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
    length = check_length(times=times)

    # [2024] STD: TC      | CAPPI: 5 sites | Mosaic: 5 sites.
    # [2025] STD: 6 sites | CAPPI: 8 sites | Mosaic: 8 sites.
    bar = alive_it(range(length), spinner="dots", bar="brackets")
    for index in bar:
        # [2024] STD: TC.
        if year == "2024":
            bar.text("Standardization: TC")
            time = str(times["TC"][index])
            cmd = [python, "standardization.py", "-s", "TC", "-t", time]
            run(cmd, shell=False, check=True, capture_output=True)
        # [2025] STD: 6 sites.
        elif year == "2025":
            process = []
            bar.text(f"Standardization: {' | '.join(standardization)}")
            for site in standardization:
                time = str(times[site][index])
                cmd = [python, "standardization.py", "-s", site, "-t", time]
                proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)
                process.append((proc, site))
            check_proc(process, standardization.copy(), bar=bar, mode="S")

        # [2024] CAPPI: 5 sites.
        # [2025] CAPPI: 8 sites.
        process = []
        bar.text(f"CAPPI: {' | '.join(sites)}")
        for site in sites:
            time = str(times[site][index])
            cmd = [milab, "-b", "milab_cappi.py", "-s", site, "-t", time]
            proc = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)
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
        cmd = [python, "mosaic.py", "-t", *time, "-n", mosaic_name]
        run(cmd, shell=False, check=True, capture_output=True)


if __name__ == "__main__":
    main()
