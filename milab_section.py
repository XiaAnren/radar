"""Get CAPPI with MeteoInfoLab."""

import argparse

# AD / TC / QFL / TZ / DLD / JD / BH / ZS
site = "JD"
time = "20250302060600"

parser = argparse.ArgumentParser(
    description="Get CAPPI using MeteoInfoLab with site and time parameters.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-s",
    "--site",
    type=str,
    choices=["AD", "TC", "QFL", "TZ", "DLD", "JD", "BH", "ZS"],
    default=site,
    help="Radar site name (AD / TC / QFL / TZ / DLD / JD / BH / ZS).",
)
parser.add_argument(
    "-t",
    "--time",
    type=str,
    default=time,
    help="Time (format: YYYYMMDDHHMMSS).",
)

args = parser.parse_args()

site = args.site
time = args.time

sid = {"AD": "ZSD01", "TC":  "ZSD02", "QFL": "ZSD03",
       "TZ": "ZSD04", "DLD": "ZSD05", "JD":  "ZSD06",
       "BH": "ZLY01", "ZS":  "ZLY02"}[site]

if site in ("AD", "TC", "QFL"):
    mode = "HAXPT0164"
elif site in ("TZ", "DLD", "JD"):
    mode = ""
elif site in ("BH", "ZS"):
    mode = "AXPT0364"

filepath = "data/bzip2"
folder = "{}/{}".format(time[:8], site)
filename = "Z_RADR_I_{}_{}_O_DOR_{}_CRA_FMT.bin.bz2".format(sid, time, mode)

file = addfile(os.path.join(filepath, folder, filename))

slat = file.attrvalue("StationLatitude")[0]
slon = file.attrvalue("StationLongitude")[0]
height = file.attrvalue("AntennaHeight")[0] / 10

x = arange(-60000, 60001, 450)
y = arange(-60000, 60001, 450)
z = linspace(0, 19000, 191)

proj = geolib.AzimuthalEquidistant(central_longitude=slon, central_latitude=slat)
lon, lat = geolib.project(x, y, fromproj=proj, toproj=projinfo())

data = file.get_grid_3d_data("dBZ", x, y, z, height)

latdim = dimension(lat, "latitude", "Y")
londim = dimension(lon, "longitude", "X")
zdim = dimension(z, "height", "Z")

savepath = "data/cappi"
folder = "section-{}/{}".format(time[:8], site)
savename = "{}.nc".format(time)

ncpath = os.path.join(savepath, folder)
if not os.path.exists(ncpath):
    os.makedirs(ncpath)

ncfile = os.path.join(savepath, folder, savename)
ncwrite(
    ncfile,
    data,
    "REF",
    dims=[zdim, latdim, londim],
    attrs={"units": "dBZ"},
    gattrs={"software": "MeteoInfo", "slat": slat, "slon": slon},
)
