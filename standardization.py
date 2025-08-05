"""PhasedArrayData -> StandardData."""

import argparse
import os
from bz2 import BZ2File
from io import BytesIO
from pathlib import Path

import numpy as np
import structure
from cinrad.io import PhasedArrayData


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Namespace containing parsed command-line arguments.

    """
    # AD / TC / QFL / TZ / DLD / JD / BH / ZS
    site = "TZ"
    time = "20250302060613"

    parser = argparse.ArgumentParser(
        description=(
            "Get StandardData from corresponding PhasedArrayData "
            "based on a StandardData template with site and time parameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--site",
        type=str,
        choices=["AD", "TC", "QFL", "TZ", "DLD", "JD"],
        default=site,
        help="Radar site name (AD / TC / QFL / TZ / DLD / JD).",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default=time,
        help="Time (format: YYYYMMDDHHMMSS).",
    )
    return parser.parse_args()


def main() -> None:
    """Standardization."""
    os.chdir(Path(__file__).resolve().parent)

    args = parse_args()
    site, time = args.site, args.time

    number = ["AD", "TC", "QFL", "TZ", "DLD", "JD"].index(site) + 1
    mode= "HAXPT0164" if site in ("AD", "TC", "QFL") else ""

    filepath = Path("/data3/gengruomei/radar/radarUTC-standard")
    folder = f"{time[:8]}/{site}"
    filename = f"Z_RADR_I_ZSD{number:02}_{time}_O_DOR_{mode}_CRA_FMT.bin.bz2"

    savepath = Path(f"data/bzip2/{time[:8]}/{site}")

    template = BZ2File("data/template.bin.bz2")

    file = PhasedArrayData(filepath / folder / filename)

    data = BytesIO()

    data.write(template.read(32))

    buffer = template.read(128)
    site_config = np.frombuffer(buffer, structure.pa_site_config_dtype).copy()
    site_config["Latitude"] = np.float32(file.stationlat)
    site_config["Longitude"] = np.float32(file.stationlon)
    height = np.int32(file.radarheight * 10)
    site_config["ground_height"] = site_config["antenna_height"] = height
    data.write(site_config.tobytes())

    buffer = template.read(256)
    task = np.frombuffer(buffer, structure.pa_task_config_dtype)
    data.write(buffer)

    san_beam_number = task["san_beam_number"][0]
    data.write(template.read(san_beam_number * 640))

    cut_num = task["cut_number"][0]
    data.write(template.read(256 * cut_num))

    # 360 azimuth -> 43 elevation -> 10 products.
    code = 2
    azimuth, elevation = 0, 42
    while True:
        buffer = template.read(128)
        data.write(buffer)
        if not buffer:
            break
        radial_header = np.frombuffer(buffer, structure.pa_radial_header_dtype)
        el_num = radial_header["elevation_number"][0] - 1
        # 10 products.
        for _ in range(radial_header["moment_number"][0]):
            buffer = template.read(32)
            moment_header = np.frombuffer(
                buffer,
                structure.moment_header_dtype,
            )
            data.write(buffer)
            dtype_code = moment_header["data_type"][0]
            # dtype = "u1".
            buffer = template.read(moment_header["block_length"][0])
            if dtype_code == code:
                buffer = (
                    (
                        file.data[el_num]["REF"][
                            azimuth,
                            : moment_header["block_length"][0],
                        ]
                        * moment_header["scale"][0]
                        + moment_header["offset"][0]
                    )
                    .filled(0)
                    .astype(f"u{moment_header['bin_length'][0]}")
                    .tobytes()
                )
                azimuth += 1 if el_num == elevation else 0
            data.write(buffer)

    savepath.mkdir(parents=True, exist_ok=True)
    with BZ2File(savepath / filename, "wb") as file:
        data.seek(0)
        file.write(data.read())


if __name__ == "__main__":
    main()
